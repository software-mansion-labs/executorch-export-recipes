"""Echo -> ExecuTorch MLX (multi_pte, Apple GPU, physical devices only), the programs react-native-executorch ships.

  reverb_mlx_{fp16,int8,int4}.pte  backbone: prefill / step / polish (_echo/v3.py), one fp16 cache

iOS pairs them with the Core ML codec files (coreml/).

fp16 comes from the fp32 checkpoint, int8 and int4 from w8a16 and w4a16, whose own codes run in MLX's quantized
matmul. On an iPhone 17 fp16 is 1.6-2x faster than both; on an M4 Pro int8 and int4 are 5-10% ahead (NOTES.md).

Needs executorch from the software-mansion-labs fork (see pyproject.toml here).
"""

import importlib.util
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent.parent
_PKG = "echo_mlx_echo"

_CHECKPOINTS = {"fp16": "fp32", "int8": "w8a16", "int4": "w4a16"}


def _vendored():
    if _PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _PKG, _MODEL_DIR / "_echo" / "__init__.py", submodule_search_locations=[str(_MODEL_DIR / "_echo")]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_PKG] = mod
        spec.loader.exec_module(mod)
    return tuple(importlib.import_module(f"{_PKG}.{m}") for m in ("hub", "v3"))


def _program(variant: dict) -> bytes:
    hub, v3 = _vendored()
    echo = v3.Echo(hub.checkpoint(_CHECKPOINTS[variant["precision"]]))
    return v3.lower(v3.export(echo), v3.schema(echo))


def export(config: dict, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in config["variants"]:
        with torch.no_grad():
            program = _program(variant)
        out_path = out_dir / variant["file"]
        out_path.write_bytes(program)
        written.append(str(out_path))
        del program
    return written
