import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _apply_patches():
    spec = importlib.util.spec_from_file_location("rfdetr_det_patches", _MODEL_DIR / "_patches.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_patches()


class _RFDETRDetectionWrapper(nn.Module):
    def __init__(self, model: nn.Module, resolution: int):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def forward(self, x: torch.Tensor):
        pred_boxes, pred_logits = self.model(x)
        scores, labels = pred_logits[0].sigmoid().max(dim=-1)
        boxes = pred_boxes[0]
        cxy = boxes[..., :2]
        wh = boxes[..., 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        bboxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution
        return bboxes, scores, labels.float()


def _detach_propagated_constants(ep):
    """Detaches the tensors `constant_prop_pass` leaves attached to autograd.

    A constant shared by two partitions is deepcopied into each, and deepcopy
    refuses a non-leaf tensor, so the lowering dies well after the pass that
    caused it. torch.export also warns that they "require gradient".
    """
    for key, value in list(ep.constants.items()):
        if isinstance(value, torch.Tensor) and (
            value.requires_grad or value.grad_fn is not None
        ):
            ep.constants[key] = value.detach().clone()
    for key, value in list(ep.state_dict.items()):
        if isinstance(value, torch.nn.Parameter) and value.grad_fn is not None:
            ep.state_dict[key] = torch.nn.Parameter(
                value.detach().clone(), requires_grad=False
            )
    return ep


def _lower(exported):
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    return to_edge_transform_and_lower(
        exported,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=[VulkanPartitioner(compile_options={"force_fp16": True})],
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
    ).to_executorch()


def export(config: dict, out_dir: str) -> str:
    _apply_patches()
    from rfdetr import RFDETRNano
    from executorch.exir.passes.constant_prop_pass import constant_prop_pass

    variant = config["variant"]
    model = RFDETRNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    resolution = model.model.resolution

    wrapper = _RFDETRDetectionWrapper(actual, resolution).eval()
    example = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)

    # no_grad, not inference_mode: inference-mode tensors are rejected during
    # Vulkan lowering even though XNNPACK accepts them.
    with torch.no_grad():
        ep = torch_export(wrapper, args=example, strict=False)
        # Re-trace so every fake tensor shares one FakeTensorMode. strict=False
        # leaves the batch norm parameters under a second mode, and Vulkan's
        # FuseBatchNormPass then trips detect_fake_mode.
        ep = ep.run_decompositions({})

    # Fold the arange-built positional grids. They are recomputed every
    # inference otherwise, and they also form a tiny delegate partition that
    # trips Vulkan's per-pass fake-mode check.
    ep = _detach_propagated_constants(constant_prop_pass(ep))

    program = _lower(ep)
    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
