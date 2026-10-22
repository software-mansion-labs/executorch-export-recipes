"""Compare exported DA3 against eager and V2 on identical single-image inputs."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch,
    _unsafe_reset_threadpool,
)
from executorch.exir._serialize._program import deserialize_pte_binary


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def input_tensor(path):
    image = Image.open(path).convert("RGB").resize((518, 518), Image.Resampling.BICUBIC)
    a = np.asarray(image).copy().astype(np.float32) / 255
    a = (a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).contiguous(), image


def graph(path, expected_backend="XnnpackBackend"):
    plan = deserialize_pte_binary(path.read_bytes()).program.execution_plan[0]

    def describe(index):
        value = plan.values[index].val
        assert int(value.scalar_type) == 6
        return {"shape": list(value.sizes), "dtype": "float32"}

    contract = {
        "inputs": [describe(i) for i in plan.inputs],
        "outputs": [describe(i) for i in plan.outputs],
    }
    assert contract == {
        "inputs": [{"shape": [1, 3, 518, 518], "dtype": "float32"}],
        "outputs": [{"shape": [1, 518, 518], "dtype": "float32"}],
    }
    assert set(d.id for d in plan.delegates) == {expected_backend}
    return {
        "contract": contract,
        "bytes": path.stat().st_size,
        "delegates": len(plan.delegates),
        "instructions": dict(
            Counter(
                type(i.instr_args).__name__ for c in plan.chains for i in c.instructions
            )
        ),
        "operators": [o.name + "." + o.overload for o in plan.operators],
    }


def heatmap(values, inverse=False):
    v = np.asarray(values, dtype=np.float32).squeeze()
    if inverse:
        v = 1 / np.maximum(v, 1e-6)
    lo, hi = np.percentile(v, [2, 98])
    v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    # Common near-bright scale; separate percentile limits accommodate arbitrary depth scales.
    stops = np.array(
        [[15, 23, 60], [0, 130, 190], [80, 220, 190], [255, 235, 90]], float
    )
    t = v * 3
    i = np.minimum(t.astype(int), 2)
    f = (t - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    return Image.fromarray(np.uint8(rgb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--v3-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=15)
    args = parser.parse_args()
    torch.set_num_threads(4)
    _unsafe_reset_threadpool(4)
    torch.manual_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "v2_fp32": args.v2_dir / "xnnpack/depth_anything_v2_small_xnnpack_fp32.pte",
        "v2_int8": args.v2_dir / "optimized/depth_anything_v2_small_xnnpack_int8.pte",
        "v3_fp32": args.v3_dir / "xnnpack/depth_anything_v3_small_xnnpack_fp32.pte",
    }
    optional = args.v3_dir / "xnnpack/depth_anything_v3_small_xnnpack_int8.pte"
    if optional.exists():
        paths["v3_int8"] = optional
    models = {k: _load_for_executorch(str(p)) for k, p in paths.items()}
    source = Path(__file__).resolve().parent
    eager3 = module(source / "_model.py", "da3_reference").build_model()
    columns = 5 if "v3_int8" in models else 4
    cases = []
    figure = Image.new("RGB", (518 * columns, 560 * len(args.image)), "#101828")
    draw = ImageDraw.Draw(figure)
    with torch.inference_mode():
        for row, path in enumerate(args.image):
            x, image = input_tensor(path)
            reference = eager3(x).clone()
            outputs = {name: m.forward([x])[0].clone() for name, m in models.items()}
            assert tuple(reference.shape) == (1, 518, 518)
            metrics = {}
            for name in [k for k in outputs if k.startswith("v3")]:
                y = outputs[name]
                assert (
                    y.shape == reference.shape
                    and y.dtype == torch.float32
                    and torch.isfinite(y).all()
                )
                error = y - reference
                rmse = error.square().mean().sqrt().item()
                span = (reference.max() - reference.min()).item()
                metrics[name] = {
                    "rmse": rmse,
                    "range_normalized_rmse": rmse / max(span, 1e-9),
                    "max_abs_error": error.abs().max().item(),
                    "correlation": torch.corrcoef(
                        torch.stack([y.flatten(), reference.flatten()])
                    )[0, 1].item(),
                    "min": y.min().item(),
                    "max": y.max().item(),
                }
            a = outputs["v2_fp32"].flatten()
            b = outputs["v3_fp32"].clamp_min(1e-6).reciprocal().flatten()
            agreement = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            cases.append(
                {
                    "image": str(path),
                    "v3_export_parity": metrics,
                    "v2_vs_reciprocal_v3_correlation": agreement,
                }
            )
            panels = [
                image,
                heatmap(outputs["v2_fp32"].numpy()),
                heatmap(outputs["v2_int8"].numpy()),
                heatmap(outputs["v3_fp32"].numpy(), True),
            ]
            labels = [
                "Image",
                "V2 Small FP32",
                "V2 Small INT8",
                "V3 Small FP32 (inverse for display)",
            ]
            if "v3_int8" in outputs:
                panels.append(heatmap(outputs["v3_int8"].numpy(), True))
                labels.append("V3 Small INT8 (inverse for display)")
            for col, (panel, label) in enumerate(zip(panels, labels)):
                figure.paste(panel, (col * 518, row * 560 + 36))
                draw.text(
                    (col * 518 + 8, row * 560 + 8),
                    f"{path.stem}: {label}",
                    fill="white",
                )
            torch.save(
                {"input": x, "v3_reference": reference, **outputs},
                args.output_dir / f"{path.stem}.pt",
            )
            print(json.dumps(cases[-1]), flush=True)
        figure.save(args.output_dir / "v2-v3-depth-comparison.png")
        x, _ = input_tensor(args.image[0])
        for m in models.values():
            for _ in range(2):
                m.forward([x])
        timings = {k: [] for k in models}
        names = list(models)
        for step in range(args.runs):
            order = names[step % len(names) :] + names[: step % len(names)]
            for name in order:
                start = time.perf_counter()
                models[name].forward([x])
                timings[name].append((time.perf_counter() - start) * 1000)
        report = {
            "threads": 4,
            "input_shape": [1, 3, 518, 518],
            "runs": args.runs,
            "cases": cases,
            "models": {
                k: {
                    "path": str(p),
                    "median_ms": statistics.median(timings[k]),
                    "timings_ms": timings[k],
                    **graph(p),
                }
                for k, p in paths.items()
            },
            "note": "Same stretched square ImageNet-normalized RGB input. Visuals use 2nd–98th percentiles per model, with reciprocal V3 for common near-bright display. Cross-model correlation is agreement, not accuracy; no ground-truth depth.",
        }
        (args.output_dir / "comparison.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                {k: round(v["median_ms"], 2) for k, v in report["models"].items()}
            ),
            flush=True,
        )
        if any(
            c["v3_export_parity"]["v3_fp32"]["range_normalized_rmse"] > 1e-4
            for c in cases
        ):
            raise RuntimeError("FP32 export parity failed")


if __name__ == "__main__":
    main()
