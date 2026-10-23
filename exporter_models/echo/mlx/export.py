"""Echo -> ExecuTorch MLX (multi_pte, Apple GPU, physical devices only).

  echo_mlx_fp16.pte          backbone: encode_prefix + denoise, fp16 compute, fp32 I/O
  codec_decoder_mlx_fp32.pte decode: codes [8, T] -> audio [T * 960]

RMSNorm stays aten.rms_norm ("native"): the residual stream reaches |x| ~ 1.1e4,
so a decomposed x * x overflows fp16, while MLX's fused kernel does not.
Needs executorch from the software-mansion-labs fork (see pyproject.toml here).
"""

import importlib.util
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent.parent
_PKG = "echo_mlx_echo"


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
    backbone, codec, hub = _vendored()
    name = variant["file"]
    if name.startswith("echo_"):
        model = backbone.load(hub.checkpoint("fp32"), rms_norm="native", dtype=torch.float16)
        return backbone.export(model, dtype=torch.float16)
    if name.startswith("codec_decoder_"):
        return {"decode": codec.export_decoder(codec.load_codec(hub.codec()))}
    raise ValueError(f"unknown Echo program for variant file '{name}'")


def export(config: dict, out_dir: str):
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in config["variants"]:
        with torch.no_grad():
            eps = _programs(variant)
            program = to_edge_transform_and_lower(
                eps,
                partitioner=[MLXPartitioner()],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
                transform_passes=get_default_passes(),
            ).to_executorch()
        out_path = out_dir / variant["file"]
        out_path.write_bytes(program.buffer)
        written.append(str(out_path))
        del eps, program
    return written
