import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "exporter_models/zipdepth")
from compare import graph, load_source, parity, rgb
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch,
    _unsafe_reset_threadpool,
)

torch.set_num_threads(4)
_unsafe_reset_threadpool(4)
root = Path("/private/tmp/zipdepth")
report = {"graphs": {}}
for backend, precision, delegate in [
    ("coreml", "fp32", "CoreMLBackend"),
    ("vulkan", "fp16", "VulkanBackend"),
]:
    p = root / backend / f"zipdepth_base_384_{backend}_{precision}.pte"
    report["graphs"][backend] = graph(p, delegate, 384)
model = _load_for_executorch(str(root / "coreml/zipdepth_base_384_coreml_fp32.pte"))
ref = load_source(Path("exporter_models/zipdepth/_model.py")).build_model(fuse=False)
cases = json.loads(
    Path("/private/tmp/depth_anything_v3/comparison/comparison.json").read_text()
)["cases"]
report["cases"] = []
with torch.inference_mode():
    for case in cases:
        x = rgb(Path(case["image"]), 384)
        y = model.forward([x])[0].clone()
        check = parity(y, ref(x))
        report["cases"].append({"image": case["image"], **check})
        print(report["cases"][-1], flush=True)
    for _ in range(3):
        model.forward([x])
    times = []
    for _ in range(15):
        start = time.perf_counter()
        model.forward([x])
        times.append((time.perf_counter() - start) * 1000)
report["coreml_median_ms"] = statistics.median(times)
report["passed"] = all(
    c["range_normalized_rmse"] < 0.01 and c["correlation"] > 0.99
    for c in report["cases"]
)
(root / "backend_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
assert report["passed"]
