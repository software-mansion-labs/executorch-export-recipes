"""Re-points the existing arms at a different image.

The `.pte` arms do not depend on the input, so switching images only needs a new
input tensor and a fresh eager reference. That keeps a new image to about a
minute instead of rebuilding three 62 MB models.

Usage:
    uv run --project exporter_models/rfdetr_nano_segmentation \
        python tools/rfdetr_seg_new_input.py --image /path/to/photo.jpg
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
_MODEL_DIR = _REPO / "exporter_models" / "rfdetr_nano_segmentation"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _apply_patches():
    spec = importlib.util.spec_from_file_location("rfdetr_seg_patches", _MODEL_DIR / "_patches.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_patches()


class Wrapper(nn.Module):
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
        scores = torch.stack([max_scores, class_ids.float()], dim=-1)
        return bboxes, scores, pred_masks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default=str(_REPO / "exports" / "rfdetr-seg-arms"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    from PIL import Image

    _apply_patches()
    from rfdetr import RFDETRSegNano

    model = RFDETRSegNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    resolution = model.model.resolution
    eager = Wrapper(actual, resolution).eval()

    img = Image.open(args.image).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    x = arr.transpose(2, 0, 1)[None].astype(np.float32)
    (out_dir / "input_0.f32.bin").write_bytes(x.tobytes())
    manifest["inputs"] = [
        {"file": "input_0.f32.bin", "shape": list(x.shape), "dtype": "float32"}
    ]
    print(f"[input] {args.image} -> input_0.f32.bin {list(x.shape)}", flush=True)

    with torch.inference_mode():
        boxes, scores, masks = (t.numpy() for t in eager(torch.from_numpy(x)))
    manifest["eager"] = []
    for label, array in (("boxes", boxes), ("scores", scores), ("masks", masks)):
        name = f"eager_{label}.f32.bin"
        (out_dir / name).write_bytes(array.astype(np.float32).tobytes())
        manifest["eager"].append({"name": label, "file": name, "shape": list(array.shape)})
        print(f"[eager] {name} {list(array.shape)}", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    keep = np.nonzero(scores[0, :, 0] >= 0.3)[0]
    print(f"[eager] {len(keep)} detections over 0.3:")
    for q in keep:
        print(
            f"    q{q:3d} cls {int(scores[0, q, 1]):3d} score {scores[0, q, 0]:.3f} "
            f"box {[round(float(v), 1) for v in boxes[0, q]]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
