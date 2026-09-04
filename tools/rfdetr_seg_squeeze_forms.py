"""Tests whether a different way of dropping the batch dim avoids the collapse.

The contract bisect showed that batch-squeezing the delegate's outputs with
`x[0]` (a rank-reducing select) is what breaks Core ML's CPU path, while the
same graph with the batch dim kept is correct. The task contract still wants
rank-reduced outputs, so the question is whether another spelling of the same
reduction survives.

Both variants below emit exactly the shipped contract - (100,4), (100), (100),
(100,78,78) - and differ only in how the leading 1 is removed.
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


class _Base(nn.Module):
    def __init__(self, model, resolution):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def _common(self, x):
        pred_boxes, pred_logits, pred_masks = self.model(x)
        cxy = pred_boxes[..., :2]
        wh = pred_boxes[..., 2:].clamp(min=0.0)
        half = 0.5 * wh
        bboxes = torch.cat([cxy - half, cxy + half], dim=-1) * self.resolution
        probs = pred_logits.sigmoid()
        s, c = probs.max(dim=-1)
        return bboxes, s, c.float(), pred_masks


class ReshapeSqueeze(_Base):
    """Shipped contract, batch dropped with reshape instead of select."""

    def forward(self, x):
        b, s, c, m = self._common(x)
        n = b.shape[1]
        return (
            b.reshape(n, 4),
            s.reshape(n),
            c.reshape(n),
            m.reshape(n, m.shape[2], m.shape[3]),
        )


class SqueezeDim0(_Base):
    """Shipped contract, batch dropped with squeeze(0)."""

    def forward(self, x):
        b, s, c, m = self._common(x)
        return b.squeeze(0), s.squeeze(0), c.squeeze(0), m.squeeze(0)


VARIANTS = {"reshape": ReshapeSqueeze, "squeezedim0": SqueezeDim0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default=str(_REPO / "exports" / "rfdetr-seg-squeeze-forms"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import coremltools as ct
    from PIL import Image
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    _apply_patches()
    from rfdetr import RFDETRSegNano

    manifest = {"arms": [], "inputs": [], "eager": []}
    resolution = None

    for name, cls in VARIANTS.items():
        print(f"[squeeze] building {name} on CPU_ONLY", flush=True)
        model = RFDETRSegNano()
        actual = model.model.model.cpu()
        actual.eval()
        actual.export()
        resolution = model.model.resolution
        wrapper = cls(actual, resolution).eval()

        example = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)
        with torch.inference_mode():
            ep = torch.export.export(wrapper, args=example, strict=False)

        specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18,
            compute_unit=ct.ComputeUnit.CPU_ONLY,
            compute_precision=ct.precision.FLOAT16,
        )
        lowered = to_edge_transform_and_lower(
            ep,
            transform_passes=[AddmmToLinearTransform()],
            partitioner=[CoreMLPartitioner(compile_specs=specs, lower_full_graph=False)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).to_executorch()

        path = out_dir / f"rfdetr_seg_{name}_cpu_only.pte"
        path.write_bytes(lowered.buffer)
        print(f"[squeeze] wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        manifest["arms"].append({"id": name, "computeUnit": "cpu_only", "file": path.name})

    img = Image.open(args.image).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    x = arr.transpose(2, 0, 1)[None].astype(np.float32)
    (out_dir / "input_0.f32.bin").write_bytes(x.tobytes())
    manifest["inputs"].append({"file": "input_0.f32.bin", "shape": list(x.shape), "dtype": "float32"})

    model = RFDETRSegNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    ref = ReshapeSqueeze(actual, resolution).eval()
    with torch.inference_mode():
        b, s, c, m = (t.numpy() for t in ref(torch.from_numpy(x)))
    for label, array in (("boxes", b), ("scores", s), ("labels", c), ("masks", m)):
        nm = f"eager_{label}.f32.bin"
        (out_dir / nm).write_bytes(array.astype(np.float32).tobytes())
        manifest["eager"].append({"name": label, "file": nm, "shape": list(array.shape)})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("[squeeze] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
