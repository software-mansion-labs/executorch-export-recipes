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


def _lower(exported, precision: str):
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    # fp16, not int8. The previous int8 build was weight-only quantization on top
    # of compute_precision=FLOAT32, which got the accuracy cost of quantization
    # and none of the benefit: fp32 compute is the one precision the ANE refuses,
    # so the model was locked onto CPU/GPU. Measured on an iPhone 16 (3 warmup +
    # 15 timed, median):
    #   detector      int8 33.2 ms -> fp16  8.2 ms   4.0x
    #   segmentation  int8 50.3 ms -> fp16 31.0 ms   1.6x
    # int8 was also measurably less accurate: confident detections (score > 0.3)
    # agreed with fp16 on only 71.9% of boxes at IoU 0.9 for the detector. fp16
    # costs about 2x the file size (27.7 -> 54.7 MB detector).
    quant = {"mode": "linear_symmetric", "dtype": "int8", "granularity": "per_channel"}
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=(
            ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
        ),
        op_linear_quantizer_config=quant if precision == "int8" else None,
    )
    # lower_full_graph=False lets the partitioner leave ops CoreML can't take
    # (e.g. a rank-6 reshape) on the CPU backend instead of failing the export.
    return to_edge_transform_and_lower(
        exported,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=[CoreMLPartitioner(compile_specs=specs, lower_full_graph=False)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
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
    example = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)
    with torch.inference_mode():
        ep = torch_export(wrapper, args=example, strict=False)

    program = _lower(ep, variant["precision"])
    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
