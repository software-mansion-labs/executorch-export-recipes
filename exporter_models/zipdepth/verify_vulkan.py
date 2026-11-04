import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "exporter_models/zipdepth")
from compare import load_source, parity, rgb

p = Path(__file__).resolve().parent / "vulkan/export.py"
s = importlib.util.spec_from_file_location("zipvk", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
torch.set_num_threads(4)
reference = load_source(Path("exporter_models/zipdepth/_model.py")).build_model(
    fuse=False
)
adapted = m._load_model()
cases = []
with torch.inference_mode():
    for case in json.loads(
        Path("/private/tmp/depth_anything_v3/comparison/comparison.json").read_text()
    )["cases"]:
        photo = Path(case["image"])
        x = rgb(photo, 384)
        check = parity(adapted(x), reference(x))
        cases.append({"image": str(photo), **check})
        assert check["range_normalized_rmse"] < 1e-4
Path("/private/tmp/zipdepth/vulkan_adaptation_verification.json").write_text(
    json.dumps({"passed": True, "cases": cases}, indent=2) + "\n"
)
print(cases, flush=True)
