"""Echo -> ExecuTorch Vulkan (multi_pte, Android GPU).

  echo_vulkan_fp16.pte   backbone: prefill / step / polish (_echo/v3.py, the MLX program's contract), fp16 compute

From the fp32 checkpoint. No codec here: Vulkan cannot take the decoder in one blob and runs it ~8x slower than
XNNPACK, so Android pairs the backbone with the XNNPACK codec files. Each method is one Vulkan delegate; the runtime
needs llama.custom_sdpa with an attn_mask (react-native-executorch's native libs v0.11.1 or later). Needs executorch
from the software-mansion-labs fork (see pyproject.toml here and NOTES.md).
"""

import importlib.util
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent.parent
_PKG = "echo_vulkan_echo"


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
    name = variant["file"]
    if name.startswith("echo_"):
        echo = v3.VulkanEcho(hub.checkpoint("fp32"))
        return v3.lower_vulkan(v3.export_vulkan(echo), v3.schema(echo))
    raise ValueError(f"unknown Echo program for variant file '{name}'")


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
