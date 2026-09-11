import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_PKG = _MODEL_DIR / "_facemesh"


def _load_facemesh():
    spec = importlib.util.spec_from_file_location(
        "facemesh_pkg_xnnpack",
        _PKG / "__init__.py",
        submodule_search_locations=[str(_PKG)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["facemesh_pkg_xnnpack"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from torch.export import export as torch_export

    from exporter.backends import lower

    fm = _load_facemesh()
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    model = fm.build()
    inputs = (torch.randn(*shape, dtype=torch.float32),)

    ep = torch_export(model, inputs, strict=True)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
