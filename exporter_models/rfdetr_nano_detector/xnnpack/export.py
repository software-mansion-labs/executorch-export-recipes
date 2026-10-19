import importlib.util
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


def _quantize_linears(module: nn.Module, scheme: str):
    """Dynamic int8 activations + int8 per-channel weights on every Linear (convs stay fp32)."""
    from torchao.quantization.granularity import PerAxis
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_

    assert scheme == "8da8w-pc", scheme
    qcfg = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int8, weight_granularity=PerAxis(0)
    )
    # exact type: MultiheadAttention's out_proj subclass is used through its raw weight
    quantize_(module, qcfg, filter_fn=lambda mod, fqn: type(mod) is nn.Linear)


def _lower(exported, backend: str):
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    return to_edge_transform_and_lower(
        exported,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
    ).to_executorch()


def export(config: dict, out_dir: str) -> str:
    _apply_patches()
    from rfdetr import RFDETRNano

    variant = config["variant"]
    model = RFDETRNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    resolution = model.model.resolution

    wrapper = _RFDETRDetectionWrapper(actual, resolution).eval()
    if variant.get("quantize"):
        _quantize_linears(wrapper, variant["quantize"])
    example = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)
    # torchao's quantized weights cannot be traced under inference_mode
    grad_ctx = torch.no_grad if variant.get("quantize") else torch.inference_mode
    with grad_ctx():
        ep = torch_export(wrapper, args=example, strict=False)

    program = _lower(ep, config["backend"])
    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
