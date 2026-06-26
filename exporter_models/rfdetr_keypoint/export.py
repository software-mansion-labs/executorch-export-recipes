import importlib
import importlib.util
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent


def _apply_patches():
    spec = importlib.util.spec_from_file_location("rfdetr_kp_patches", _HERE / "_patches.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_patches()


class _RFDETRKeypointWrapper(nn.Module):
    def __init__(self, model: nn.Module, resolution: int, person_class_index: int = 1, num_keypoints: int = 17):
        super().__init__()
        self.model = model
        self.resolution = resolution
        self.person_class_index = person_class_index
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor):
        forward_fn = getattr(self.model, "forward_export", self.model)
        pred_boxes, pred_logits, pred_keypoints = forward_fn(x)
        cxy = pred_boxes[0, :, :2]
        wh = pred_boxes[0, :, 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        boxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution
        scores = pred_logits[0, :, self.person_class_index].sigmoid()
        kp_start = self.person_class_index * self.num_keypoints
        kp_slice = pred_keypoints[0, :, kp_start : kp_start + self.num_keypoints, :]
        kp_xy = (kp_slice[..., :2] * self.resolution).clamp(min=0.0, max=float(self.resolution))
        kp_vis = kp_slice[..., 2].sigmoid().unsqueeze(-1)
        keypoints = torch.cat([kp_xy, kp_vis], dim=-1)
        return boxes, scores, keypoints


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

    variants = importlib.import_module("rfdetr.variants")
    model_cls = getattr(variants, "RFDETRKeypointPreview")
    wrapper_meta = model_cls()
    inner = deepcopy(wrapper_meta.model.model.to("cpu")).eval()
    inner.export()
    resolution = wrapper_meta.model.resolution

    variant = config["variant"]
    wrapper = _RFDETRKeypointWrapper(inner, resolution).eval()
    example = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)
    with torch.inference_mode():
        ep = torch_export(wrapper, args=example, strict=False)

    program = _lower(ep, config["backend"])
    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
