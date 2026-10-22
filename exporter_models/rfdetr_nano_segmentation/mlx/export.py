import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _apply_patches():
    spec = importlib.util.spec_from_file_location("rfdetr_seg_patches", _MODEL_DIR / "_patches.py")
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
        # The runtime InstanceSegmenter validates a 4-output, batch-free
        # signature: boxes[N,4], scores[N], labels[N], masks[N,MH,MW]. Index [0]
        # to drop the batch dim and return the four tensors separately.
        boxes = pred_boxes[0]
        cxy = boxes[..., :2]
        wh = boxes[..., 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        bboxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution
        # Split max(dim=-1) into single-output amax/argmax so the MLX builder
        # doesn't hit the multi-output getitem path (see the detector export).
        probs = pred_logits[0].sigmoid()
        scores = probs.amax(dim=-1)
        labels = probs.argmax(dim=-1).float()
        masks = pred_masks[0]
        return bboxes, scores, labels, masks


class _Fp16IO(torch.nn.Module):
    """fp32 I/O, fp16 compute — mirrors CoreML FLOAT16 semantics so the app's
    fp32 input/output contract is preserved while the graph runs in half."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args):
        args = tuple(
            a.to(torch.float16) if torch.is_floating_point(a) else a for a in args
        )
        out = self.inner(*args)
        if isinstance(out, (tuple, list)):
            return tuple(
                o.to(torch.float32) if torch.is_floating_point(o) else o for o in out
            )
        return out.to(torch.float32)


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

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
    if variant["precision"] == "fp16":
        wrapper = _Fp16IO(wrapper.half().eval()).eval()
    with torch.inference_mode():
        ep = torch_export(wrapper, args=example, strict=False)

    program = to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
