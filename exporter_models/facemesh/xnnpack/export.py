"""XNNPACK export: fp32, and only fp32.

This is the one backend that has to run everywhere, so it is also the one where
narrowing the weights is tempting. It does not pay here, and the measurements
are in NOTES.md: int8 runs at 0.78x fp32 on a Galaxy S26 Ultra and 0.80x on an
iPhone 16 Pro, while drifting 0.62 px mean / 6.3 px max and moving the
face-presence score by up to 0.494.

The reason is the shape of the graph rather than anything about the quantizer.
The whole model is ~0.75 ms of small depthwise convolutions at 192x192, so it is
bound by per-op overhead and not by arithmetic width; narrowing the arithmetic
saves nothing and the quantize/dequantize boundaries it adds cost real time.
fp16 is not an option either — XNNPACK computes in fp32 regardless, so it would
buy file size and lose accuracy.

The 2.4 MB that fp32 costs is the right trade for a model this small. If you
want it smaller, Core ML fp16 already ships at 1.5 MB on iOS.
"""

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
