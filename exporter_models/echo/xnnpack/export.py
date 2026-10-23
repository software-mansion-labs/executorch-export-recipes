"""Echo -> ExecuTorch XNNPACK (multi_pte).

  echo_xnnpack_{fp32,8da4w}.pte  backbone: encode_prefix + denoise
  codec_decoder_xnnpack_fp32.pte decode: codes [8, T] -> audio [T * 960]
  codec_encoder_xnnpack_fp32.pte encode: 10 s of 24 kHz audio -> codes [8, 250]

The backbone lowers to one XNNPACK blob per method: RMSNorm's mean is a
constant linear ("linear" mode). 8da4w is the w4a16 QAT checkpoint with its own
int4 codes injected, run with int8 dynamic activations.
"""

import importlib.util
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent.parent
_PKG = "echo_xnnpack_echo"


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
        quantized = variant["precision"] == "8da4w"
        ckpt = hub.checkpoint("w4a16" if quantized else "fp32")
        return backbone.export(backbone.load(ckpt, rms_norm="linear", dynamic_quant=quantized))
    if name.startswith("codec_decoder_"):
        return {"decode": codec.export_decoder(codec.load_codec(hub.codec()))}
    if name.startswith("codec_encoder_"):
        return {"encode": codec.export_encoder(codec.load_codec(hub.codec()), seconds=10.0)}
    raise ValueError(f"unknown Echo program for variant file '{name}'")


def export(config: dict, out_dir: str):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in config["variants"]:
        with torch.no_grad():
            eps = _programs(variant)
            program = to_edge_transform_and_lower(
                eps,
                partitioner=[XnnpackPartitioner()],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            ).to_executorch()
        out_path = out_dir / variant["file"]
        out_path.write_bytes(program.buffer)
        written.append(str(out_path))
        del eps, program
    return written
