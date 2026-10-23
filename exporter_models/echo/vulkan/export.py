"""Echo -> ExecuTorch Vulkan (multi_pte, Android GPU, fp16 compute).

  echo_vulkan_fp16.pte   backbone: encode_prefix + denoise, fp32 I/O
  echo_vulkan_8da4w.pte  same, w4a16 QAT codes with int8 dynamic activations

No codec here: Vulkan cannot take the decoder in one blob and runs it ~8x
slower than XNNPACK, so Android pairs these with codec_*_xnnpack_fp32.pte.
RMSNorm is spelled x * rsqrt(mean(x * x) + eps) ("vulkan"), which the Vulkan
FusePatternsPass folds into its fp32-accumulating rms_norm kernel; fp16 x * x
would overflow (|x| ~ 1.1e4). Needs executorch from the software-mansion-labs
fork (see pyproject.toml here and NOTES.md).
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
    return tuple(importlib.import_module(f"{_PKG}.{m}") for m in ("backbone", "codec", "hub"))


def _programs(variant: dict) -> dict:
    backbone, _, hub = _vendored()
    name = variant["file"]
    if name.startswith("echo_"):
        quantized = variant["precision"] == "8da4w"
        ckpt = hub.checkpoint("w4a16" if quantized else "fp32")
        return backbone.export(backbone.load(ckpt, rms_norm="vulkan", dynamic_quant=quantized))
    raise ValueError(f"unknown Echo program for variant file '{name}'")


def export(config: dict, out_dir: str):
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in config["variants"]:
        with torch.no_grad():
            eps = _programs(variant)
            program = to_edge_transform_and_lower(
                eps,
                partitioner=[VulkanPartitioner({"texture_limits": (2048, 2048, 2048), "force_fp16": True})],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            ).to_executorch()
        out_path = out_dir / variant["file"]
        out_path.write_bytes(program.buffer)
        written.append(str(out_path))
        del eps, program
    return written
