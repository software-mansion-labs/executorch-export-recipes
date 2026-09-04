"""Builds compute-unit arms from the SHIPPED wrapper, not a stand-in.

The first set of arms used an older three-output batched wrapper. Those arms
showed no ANE defect at all, which leaves open the possibility that the defect
lives in the graph difference rather than in the compute unit: the shipped
export emits four batch-squeezed outputs and splits scores from class ids with
a separate `max`, and a different graph can be partitioned differently and
placed differently on the ANE.

So this imports `_RFDETRSegmentationWrapper` and `_lower` from the exporter
itself and varies nothing but `compute_unit`, which is the only way the
comparison is about compute units.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
_MODEL_DIR = _REPO / "exporter_models" / "rfdetr_nano_segmentation"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_exporter():
    spec = importlib.util.spec_from_file_location(
        "rfdetr_seg_coreml_export", _MODEL_DIR / "coreml" / "export.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default=str(_REPO / "exports" / "rfdetr-seg-shipped-arms"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import coremltools as ct
    from PIL import Image
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    exporter = _load_exporter()
    exporter._apply_patches()
    from rfdetr import RFDETRSegNano

    ARMS = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }

    manifest = {"arms": [], "inputs": [], "eager": []}
    resolution = None

    for name, unit in ARMS.items():
        print(f"[arms] building {name} (shipped wrapper)", flush=True)
        model = RFDETRSegNano()
        actual = model.model.model.cpu()
        actual.eval()
        actual.export()
        resolution = model.model.resolution
        # The exporter's own wrapper: four batch-squeezed outputs.
        wrapper = exporter._RFDETRSegmentationWrapper(actual, resolution).eval()

        example = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)
        with torch.inference_mode():
            ep = torch.export.export(wrapper, args=example, strict=False)

        specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18,
            compute_unit=unit,
            compute_precision=ct.precision.FLOAT16,
        )
        lowered = to_edge_transform_and_lower(
            ep,
            transform_passes=[AddmmToLinearTransform()],
            partitioner=[CoreMLPartitioner(compile_specs=specs, lower_full_graph=False)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).to_executorch()

        path = out_dir / f"rfdetr_seg_shipped_fp16_{name}.pte"
        path.write_bytes(lowered.buffer)
        print(f"[arms] wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        manifest["arms"].append({"id": name, "computeUnit": name, "file": path.name})

    img = Image.open(args.image).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    x = arr.transpose(2, 0, 1)[None].astype(np.float32)
    (out_dir / "input_0.f32.bin").write_bytes(x.tobytes())
    manifest["inputs"].append(
        {"file": "input_0.f32.bin", "shape": list(x.shape), "dtype": "float32"}
    )
    print(f"[input] {args.image} -> input_0.f32.bin {list(x.shape)}", flush=True)

    print("[arms] eager reference", flush=True)
    model = RFDETRSegNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    eager = exporter._RFDETRSegmentationWrapper(actual, model.model.resolution).eval()
    with torch.inference_mode():
        boxes, scores, labels, masks = (t.numpy() for t in eager(torch.from_numpy(x)))
    for label, array in (
        ("boxes", boxes),
        ("scores", scores),
        ("labels", labels),
        ("masks", masks),
    ):
        name = f"eager_{label}.f32.bin"
        (out_dir / name).write_bytes(array.astype(np.float32).tobytes())
        manifest["eager"].append({"name": label, "file": name, "shape": list(array.shape)})
        print(f"[eager] {name} {list(array.shape)}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    keep = np.nonzero(scores >= 0.3)[0]
    print(f"[eager] {len(keep)} detections over 0.3:")
    for q in keep:
        print(
            f"    q{q:3d} cls {int(labels[q]):3d} score {scores[q]:.3f} "
            f"box {[round(float(v), 1) for v in boxes[q]]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
