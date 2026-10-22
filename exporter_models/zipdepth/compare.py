"""Validate ZipDepth exports and compare against V2 on saved photos and NYUv2 labels."""

import argparse
import importlib.util
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch,
    _unsafe_reset_threadpool,
)
from PIL import Image, ImageDraw


def load_source(path):
    spec = importlib.util.spec_from_file_location("zipdepth_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph(path, backend, side):
    plan = deserialize_pte_binary(path.read_bytes()).program.execution_plan[0]

    def meta(i):
        v = plan.values[i].val
        assert int(v.scalar_type) == 6
        return list(v.sizes)

    assert [meta(i) for i in plan.inputs] == [[1, 3, side, side]]
    assert [meta(i) for i in plan.outputs] == [[1, side, side]]
    assert {d.id for d in plan.delegates} == {backend}
    return {
        "bytes": path.stat().st_size,
        "input_resolution": side,
        "delegates": len(plan.delegates),
        "instructions": dict(
            Counter(
                type(i.instr_args).__name__ for c in plan.chains for i in c.instructions
            )
        ),
    }


def rgb(path, side):
    image = (
        Image.open(path).convert("RGB").resize((side, side), Image.Resampling.BILINEAR)
    )
    return (
        torch.from_numpy(np.array(image).astype(np.float32) / 255)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )


def resize(depth, size):
    return F.interpolate(
        depth.reshape(1, 1, *depth.shape[-2:]),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def heatmap(values):
    v = np.asarray(values).squeeze()
    lo, hi = np.percentile(v, [2, 98])
    v = np.clip((v - lo) / max(hi - lo, 1e-8), 0, 1)
    stops = np.array(
        [[15, 23, 60], [0, 130, 190], [80, 220, 190], [255, 235, 90]], float
    )
    t = v * 3
    i = np.minimum(t.astype(int), 2)
    f = (t - i)[..., None]
    return Image.fromarray(np.uint8(stops[i] * (1 - f) + stops[i + 1] * f))


def parity(actual, reference):
    assert (
        actual.shape == reference.shape
        and actual.dtype == torch.float32
        and torch.isfinite(actual).all()
    )
    error = actual - reference
    span = (reference.max() - reference.min()).item()
    return {
        "max_abs_error": error.abs().max().item(),
        "range_normalized_rmse": error.square().mean().sqrt().item() / max(span, 1e-9),
        "correlation": torch.corrcoef(
            torch.stack([actual.flatten(), reference.flatten()])
        )[0, 1].item(),
    }


def depth_metrics(prediction, truth):
    # Relative inverse-depth alignment: both models have arbitrary scale and shift.
    prediction = resize(prediction, truth.shape).double()
    truth = truth.double()
    valid = torch.isfinite(truth) & (truth > 0.1) & (truth < 10)
    yy, xx = torch.meshgrid(
        torch.arange(truth.shape[0]), torch.arange(truth.shape[1]), indexing="ij"
    )
    valid &= (
        (yy >= truth.shape[0] * 0.05)
        & (yy < truth.shape[0] * 0.95)
        & (xx >= truth.shape[1] * 0.05)
        & (xx < truth.shape[1] * 0.95)
    )
    p = prediction[valid]
    g = truth[valid]
    design = torch.stack([p, torch.ones_like(p)], dim=1)
    scale, shift = torch.linalg.lstsq(design, g.reciprocal()).solution
    estimated = (scale * p + shift).clamp_min(0.1).reciprocal().clamp(0.1, 10)
    return {
        "abs_rel": ((estimated - g).abs() / g).mean().item(),
        "rmse": (estimated - g).square().mean().sqrt().item(),
        "delta1": (torch.maximum(estimated / g, g / estimated) < 1.25)
        .double()
        .mean()
        .item(),
        "alignment_scale": scale.item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--nyu-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    _unsafe_reset_threadpool(4)
    paths = {
        "v2_xnnpack_int8": Path(
            "/private/tmp/depth_anything_v2/optimized/depth_anything_v2_small_xnnpack_int8.pte"
        ),
        "zip_xnnpack_fp32": args.artifacts
        / "xnnpack/zipdepth_base_384_xnnpack_fp32.pte",
        "v2_coreml_fp16": Path(
            "/private/tmp/depth_anything_v2/coreml/depth_anything_v2_small_coreml_fp16.pte"
        ),
        "zip_coreml_fp32": args.artifacts / "coreml/zipdepth_base_384_coreml_fp32.pte",
    }
    models = {name: _load_for_executorch(str(path)) for name, path in paths.items()}
    prior = json.loads((args.references / "comparison.json").read_text())
    report_path = args.output_dir / "comparison.json"
    report = (
        json.loads(report_path.read_text())
        if args.benchmark_only
        else {"cases": [], "nyu_cases": []}
    )
    with torch.inference_mode():
        if not args.benchmark_only:
            source = load_source(Path(__file__).parent / "_model.py")
            unfused = source.build_model(fuse=False)
            fused = source.build_model()
            figure = Image.new("RGB", (518 * 4, 560 * len(prior["cases"])), "#101828")
            draw = ImageDraw.Draw(figure)
            for row, case in enumerate(prior["cases"]):
                photo = Path(case["image"])
                x = rgb(photo, 384)
                expected = unfused(x).clone()
                fused_y = fused(x).clone()
                outputs = {
                    name: models[name].forward([x])[0].clone()
                    for name in ["zip_xnnpack_fp32", "zip_coreml_fp32"]
                }
                checks = {name: parity(y, expected) for name, y in outputs.items()}
                checks["fusion"] = parity(fused_y, expected)
                saved = torch.load(
                    args.references / (photo.stem + ".pt"), weights_only=True
                )
                agreement = torch.corrcoef(
                    torch.stack(
                        [
                            saved["v2_fp32"].flatten(),
                            resize(outputs["zip_xnnpack_fp32"], (518, 518)).flatten(),
                        ]
                    )
                )[0, 1].item()
                report["cases"].append(
                    {
                        "image": str(photo),
                        "parity": checks,
                        "v2_zip_correlation": agreement,
                    }
                )
                panels = [
                    Image.open(photo).convert("RGB").resize((518, 518)),
                    heatmap(saved["v2_fp32"].numpy()),
                    heatmap(outputs["zip_xnnpack_fp32"].numpy()).resize((518, 518)),
                    heatmap(outputs["zip_coreml_fp32"].numpy()).resize((518, 518)),
                ]
                for col, (panel, label) in enumerate(
                    zip(
                        panels,
                        [
                            "Image",
                            "V2 FP32 518",
                            "ZipDepth FP32 384",
                            "ZipDepth CoreML FP32 384",
                        ],
                    )
                ):
                    figure.paste(panel, (col * 518, row * 560 + 36))
                    draw.text(
                        (col * 518 + 8, row * 560 + 8),
                        photo.stem + ": " + label,
                        fill="white",
                    )
                print(json.dumps(report["cases"][-1]), flush=True)
            figure.save(args.output_dir / "v2-zipdepth-comparison.png")
            if args.nyu_dir:
                v2 = _load_for_executorch(
                    "/private/tmp/depth_anything_v2/xnnpack/depth_anything_v2_small_xnnpack_fp32.pte"
                )
                for path in sorted(args.nyu_dir.glob("nyu_*.pt")):
                    data = torch.load(path, weights_only=True)
                    truth = data["depth"][0]
                    image = data["image"].unsqueeze(0)
                    xzip = F.interpolate(
                        image, size=(384, 384), mode="bilinear", align_corners=False
                    ).contiguous()
                    xv2 = F.interpolate(
                        image, size=(518, 518), mode="bicubic", align_corners=False
                    ).clamp(0, 1)
                    xv2 = (
                        (xv2 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                        / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    ).contiguous()
                    outputs = {
                        "v2_fp32": v2.forward([xv2])[0].clone(),
                        "zip_fp32": models["zip_xnnpack_fp32"]
                        .forward([xzip])[0]
                        .clone(),
                    }
                    checks = {
                        name: depth_metrics(y, truth) for name, y in outputs.items()
                    }
                    report["nyu_cases"].append({"case": path.stem, "metrics": checks})
                    print(json.dumps(report["nyu_cases"][-1]), flush=True)
                report["nyu_mean"] = {
                    name: {
                        metric: statistics.mean(
                            c["metrics"][name][metric] for c in report["nyu_cases"]
                        )
                        for metric in ["abs_rel", "rmse", "delta1"]
                    }
                    for name in ["v2_fp32", "zip_fp32"]
                }
            report["passed"] = all(
                c["parity"]["zip_xnnpack_fp32"]["range_normalized_rmse"] < 1e-4
                and c["parity"]["fusion"]["range_normalized_rmse"] < 1e-4
                and c["parity"]["zip_coreml_fp32"]["range_normalized_rmse"] < 0.01
                and c["parity"]["zip_coreml_fp32"]["correlation"] > 0.99
                for c in report["cases"]
            )
        if args.runs:
            xzip = rgb(Path(prior["cases"][0]["image"]), 384)
            xv2 = torch.load(
                args.references / (Path(prior["cases"][0]["image"]).stem + ".pt"),
                weights_only=True,
            )["input"]
            inputs = {name: xzip if name.startswith("zip") else xv2 for name in models}
            for name, model in models.items():
                for _ in range(2):
                    model.forward([inputs[name]])
            times = {name: [] for name in models}
            names = list(models)
            for step in range(args.runs):
                for name in names[step % len(names) :] + names[: step % len(names)]:
                    start = time.perf_counter()
                    models[name].forward([inputs[name]])
                    times[name].append((time.perf_counter() - start) * 1000)
            report["models"] = {
                name: {
                    "median_ms": statistics.median(times[name]),
                    "timings_ms": times[name],
                    **graph(
                        path,
                        "CoreMLBackend" if "coreml" in name else "XnnpackBackend",
                        384 if name.startswith("zip") else 518,
                    ),
                }
                for name, path in paths.items()
            }
            report["runs"] = args.runs
        report["note"] = (
            "Small deployment probe: ZipDepth 384 vs V2 518, seven photos/illustrations and first 12 NYUv2 val rows (tanganke/nyuv2). Inverse-depth least-squares scale/shift alignment, valid GT 0.1–10m, 5% border excluded. Not an official benchmark."
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "passed": report.get("passed"),
                    "nyu_mean": report.get("nyu_mean"),
                    "latency": {
                        k: v["median_ms"] for k, v in report.get("models", {}).items()
                    },
                }
            ),
            flush=True,
        )
        if not report.get("passed"):
            raise SystemExit("ZipDepth numerical validation failed")


if __name__ == "__main__":
    main()
