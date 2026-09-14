import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _apply_patches():
    spec = importlib.util.spec_from_file_location("rfdetr_seg_vulkan_patches", _MODEL_DIR / "_patches.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_patches()


class _RFDETRSegmentationWrapper(nn.Module):
    def __init__(self, model: nn.Module, resolution: int):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def forward(self, x: torch.Tensor):
        pred_boxes, pred_logits, pred_masks = self.model(x)
        cxy = pred_boxes[..., :2]
        wh = pred_boxes[..., 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        bboxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution
        probs = pred_logits.sigmoid()
        max_scores, class_ids = probs.max(dim=-1)
        # flattened contract: 4 batch-squeezed outputs (boxes, scores, labels, masks)
        return bboxes[0], max_scores[0], class_ids[0].float(), pred_masks[0]


def _lower(exported, backend: str, precision: str = "fp16"):
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    return to_edge_transform_and_lower(
        exported,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=[
            VulkanPartitioner(
                compile_options={"force_fp16": True} if precision == "fp16" else {}
            )
        ],
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
    ).to_executorch()


def export(config: dict, out_dir: str) -> str:
    _apply_patches()
    from rfdetr import RFDETRSegNano

    variant = config["variant"]
    model = RFDETRSegNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    resolution = model.model.resolution

    wrapper = _RFDETRSegmentationWrapper(actual, resolution).eval()
    example = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)
    with torch.no_grad():
        ep = torch_export(wrapper, args=example, strict=False)
        # Re-trace so every fake tensor shares one FakeTensorMode. The
        # strict=False export leaves the batch norm parameters under a second
        # mode, and Vulkan's FuseBatchNormPass then trips detect_fake_mode.
        ep = ep.run_decompositions({})
    # Fold the input-independent subgraphs (positional grids built from arange).
    # They are recomputed every inference otherwise, and here they also form a
    # tiny delegate partition that trips Vulkan's per-pass fake-mode check.
    from executorch.exir.passes.constant_prop_pass import constant_prop_pass

    ep = constant_prop_pass(ep)

    program = _lower(ep, config["backend"], variant["precision"])
    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
