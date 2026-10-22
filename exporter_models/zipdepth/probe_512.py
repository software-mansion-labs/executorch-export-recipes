"""Check a near-resolution-matched ZipDepth 512 export against V2's 518 preset."""

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from compare import depth_metrics, graph, load_source, parity, rgb
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch,
    _unsafe_reset_threadpool,
)


def main():
    torch.set_num_threads(4)
    _unsafe_reset_threadpool(4)
    root = Path("/private/tmp/zipdepth")
    paths = {
        "v2_int8_518": Path(
            "/private/tmp/depth_anything_v2/optimized/depth_anything_v2_small_xnnpack_int8.pte"
        ),
        "zip_fp32_384": root / "xnnpack/zipdepth_base_384_xnnpack_fp32.pte",
        "zip_fp32_512": root / "xnnpack/zipdepth_base_512_xnnpack_fp32.pte",
    }
    models = {k: _load_for_executorch(str(p)) for k, p in paths.items()}
    source = load_source(Path(__file__).parent / "_model.py")
    reference = source.build_model(fuse=False)
    prior = json.loads((root / "comparison/comparison.json").read_text())
    photos = json.loads(
        Path("/private/tmp/depth_anything_v3/comparison/comparison.json").read_text()
    )["cases"]
    report = {"parity": [], "nyu_cases": []}
    with torch.inference_mode():
        for case in photos:
            x = rgb(Path(case["image"]), 512)
            expected = reference(x).clone()
            actual = models["zip_fp32_512"].forward([x])[0].clone()
            check = parity(actual, expected)
            assert check["range_normalized_rmse"] < 1e-4
            report["parity"].append({"image": case["image"], **check})
        for path in sorted((root / "nyu_probe").glob("nyu_*.pt")):
            d = torch.load(path, weights_only=True)
            x = F.interpolate(
                d["image"].unsqueeze(0),
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            ).contiguous()
            y = models["zip_fp32_512"].forward([x])[0].clone()
            check = depth_metrics(y, d["depth"][0])
            report["nyu_cases"].append({"case": path.stem, "metrics": check})
        report["nyu_mean"] = {
            k: statistics.mean(c["metrics"][k] for c in report["nyu_cases"])
            for k in ["abs_rel", "rmse", "delta1"]
        }
        report["v2_nyu_mean"] = prior["nyu_mean"]["v2_fp32"]
        report["zip384_nyu_mean"] = prior["nyu_mean"]["zip_fp32"]
        photo = Path(photos[0]["image"])
        x384 = rgb(photo, 384)
        x512 = rgb(photo, 512)
        xv2 = torch.load(
            "/private/tmp/depth_anything_v3/comparison/" + photo.stem + ".pt",
            weights_only=True,
        )["input"]
        inputs = {"v2_int8_518": xv2, "zip_fp32_384": x384, "zip_fp32_512": x512}
        for name, model in models.items():
            for _ in range(2):
                model.forward([inputs[name]])
        timings = {k: [] for k in models}
        names = list(models)
        for step in range(15):
            for name in names[step % 3 :] + names[: step % 3]:
                start = time.perf_counter()
                models[name].forward([inputs[name]])
                timings[name].append((time.perf_counter() - start) * 1000)
        report["models"] = {
            k: {
                "median_ms": statistics.median(timings[k]),
                "timings_ms": timings[k],
                **graph(
                    p,
                    "XnnpackBackend",
                    518 if k.startswith("v2") else 512 if k.endswith("512") else 384,
                ),
            }
            for k, p in paths.items()
        }
    (root / "comparison/probe_512.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "nyu_mean": report["nyu_mean"],
                "v2": report["v2_nyu_mean"],
                "latency": {k: v["median_ms"] for k, v in report["models"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
