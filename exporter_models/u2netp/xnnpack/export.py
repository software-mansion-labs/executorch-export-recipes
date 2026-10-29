import importlib.util
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent.parent


def _load_wrapper():
    key = "u2netp_wrapper"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / "_wrapper.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def export(config: dict, out_dir: str) -> str:
    from exporter.backends import lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    wrapper = _load_wrapper()
    model = wrapper.U2NetPForRNE(wrapper.load_model()).eval()
    inputs = (torch.rand(*shape, dtype=torch.float32),)
    ep = torch.export.export(model, inputs, strict=True)
    program = lower(ep, config["backend"])
    path = Path(out_dir) / variant["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(program.buffer)
    return str(path)
