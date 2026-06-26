import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

from exporter.backends import lower

_HERE = Path(__file__).resolve().parent


def _load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "fast_sam_wrapper", _HERE / "_fastsam_wrapper.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fast_sam_wrapper"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from ultralytics import FastSAM

    wrapper_mod = _load_wrapper()
    FastSAMInstanceSegmentation = wrapper_mod.FastSAMInstanceSegmentation

    size = config.get("size") or "s"
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])  # [1,3,640,640]
    method_size = shape[-1]

    inner = FastSAM(f"FastSAM-{size}.pt").model
    model = FastSAMInstanceSegmentation(inner, method_size=method_size)
    model.eval()

    inputs = (torch.randn(*shape, dtype=torch.float32),)
    with torch.no_grad():
        model(*inputs)  # warmup: FastSAM head needs a forward to fix export-mode shapes

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
