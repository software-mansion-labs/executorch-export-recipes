"""Shared RF-DETR keypoint (preview) model build + export wrapper.

Backend export.py files build the model via build_wrapper() and differ only in
the lowering step.
"""
import importlib
import importlib.util
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

_MODEL_DIR = Path(__file__).resolve().parent


def apply_patches(**flags):
    """Apply the export patches. ``flags`` are forwarded to _patches.apply_patches
    so a backend can opt into its own set (the MLX path needs the grid-free
    deformable-attention core and the compile-safe positional embedding)."""
    spec = importlib.util.spec_from_file_location(
        "rfdetr_kp_patches", _MODEL_DIR / "_patches.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_patches(**flags)


class RFDETRKeypointWrapper(nn.Module):
    def __init__(self, model, resolution, person_class_index=1, num_keypoints=17):
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


def build_wrapper(patch_flags=None):
    """Returns (wrapper, example_inputs) ready for torch.export."""
    apply_patches(**(patch_flags or {}))
    variants = importlib.import_module("rfdetr.variants")
    wrapper_meta = variants.RFDETRKeypointPreview()
    inner = deepcopy(wrapper_meta.model.model.to("cpu")).eval()
    inner.export()
    # LWDETR.export() swaps forward for an instance attribute (forward_export);
    # that instance swap breaks run_decompositions on the export graph. The
    # wrapper calls forward_export directly, so restore the original forward.
    if hasattr(inner, "_forward_origin"):
        inner.forward = inner._forward_origin
    resolution = wrapper_meta.model.resolution
    wrapper = RFDETRKeypointWrapper(inner, resolution).eval()
    example = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)
    return wrapper, example
