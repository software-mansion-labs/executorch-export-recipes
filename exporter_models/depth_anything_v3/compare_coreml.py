"""Compare V2/V3 CoreML FP16 using the saved inputs and float references."""

import argparse
import json
from pathlib import Path
import statistics
import time

from PIL import Image, ImageDraw
import torch
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch,
    _unsafe_reset_threadpool,
)

from compare import graph, heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-model", type=Path, required=True)
    parser.add_argument("--v3-model", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=15)
    args = parser.parse_args()
    torch.set_num_threads(4)
    _unsafe_reset_threadpool(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior = json.loads((args.references / "comparison.json").read_text())
    paths = {"v2_coreml_fp16": args.v2_model, "v3_coreml_fp16": args.v3_model}
    graph_info = {name: graph(path, "CoreMLBackend") for name, path in paths.items()}
    models = {name: _load_for_executorch(str(path)) for name, path in paths.items()}
    cases = []
    figure = Image.new("RGB", (518 * 5, 560 * len(prior["cases"])), "#101828")
    draw = ImageDraw.Draw(figure)
    benchmark_input = None
    with torch.inference_mode():
        for row, case in enumerate(prior["cases"]):
            source = Path(case["image"])
            saved = torch.load(
                args.references / (source.stem + ".pt"), weights_only=True
            )
            x = saved["input"].contiguous()
            if benchmark_input is None:
                benchmark_input = x
            references = {
                "v2_coreml_fp16": saved["v2_fp32"],
                "v3_coreml_fp16": saved["v3_reference"],
            }
            outputs = {}
            metrics = {}
            for name, model in models.items():
                start = time.perf_counter()
                output = model.forward([x])[0].clone()
                first_ms = (time.perf_counter() - start) * 1000
                reference = references[name]
                assert output.dtype == torch.float32 and tuple(output.shape) == (
                    1,
                    518,
                    518,
                )
                assert torch.isfinite(output).all()
                error = output - reference
                nrmse = error.square().mean().sqrt().item() / max(
                    (reference.max() - reference.min()).item(), 1e-9
                )
                correlation = torch.corrcoef(
                    torch.stack([output.flatten(), reference.flatten()])
                )[0, 1].item()
                metrics[name] = {
                    "range_normalized_rmse": nrmse,
                    "max_abs_error": error.abs().max().item(),
                    "correlation": correlation,
                    "first_ms": first_ms,
                    "passed": nrmse < 0.01 and correlation > 0.99,
                }
                outputs[name] = output
            cases.append({"image": str(source), "parity": metrics})
            panels = [
                Image.open(source)
                .convert("RGB")
                .resize((518, 518), Image.Resampling.BICUBIC),
                heatmap(saved["v2_fp32"].numpy()),
                heatmap(outputs["v2_coreml_fp16"].numpy()),
                heatmap(saved["v3_fp32"].numpy(), True),
                heatmap(outputs["v3_coreml_fp16"].numpy(), True),
            ]
            labels = [
                "Image",
                "V2 XNNPACK FP32",
                "V2 CoreML FP16",
                "V3 XNNPACK FP32 (inverse)",
                "V3 CoreML FP16 (inverse)",
            ]
            for col, (panel, label) in enumerate(zip(panels, labels)):
                figure.paste(panel, (col * 518, row * 560 + 36))
                draw.text(
                    (col * 518 + 8, row * 560 + 8),
                    f"{source.stem}: {label}",
                    fill="white",
                )
            torch.save({"input": x, **outputs}, args.output_dir / (source.stem + ".pt"))
            print(json.dumps(cases[-1]), flush=True)
        figure.save(args.output_dir / "v2-v3-coreml-comparison.png")
        for model in models.values():
            for _ in range(2):
                model.forward([benchmark_input])
        times = {name: [] for name in models}
        for step in range(args.runs):
            order = list(models) if step % 2 == 0 else list(reversed(models))
            for name in order:
                start = time.perf_counter()
                models[name].forward([benchmark_input])
                times[name].append((time.perf_counter() - start) * 1000)
    report = {
        "runs": args.runs,
        "threads": 4,
        "compute_unit": "ALL",
        "precision": "fp16",
        "cases": cases,
        "models": {
            name: {
                "path": str(path),
                **graph_info[name],
                "median_ms": statistics.median(times[name]),
                "timings_ms": times[name],
            }
            for name, path in paths.items()
        },
        "passed": all(m["passed"] for c in cases for m in c["parity"].values()),
        "note": "macOS native CoreML delegate, same 518-square inputs as XNNPACK comparison. V2 reference is validated XNNPACK FP32; V3 reference is unmodified eager PyTorch. No ground-truth depth accuracy measured.",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {name: data["median_ms"] for name, data in report["models"].items()}
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit("CoreML numerical validation failed")


if __name__ == "__main__":
    main()
