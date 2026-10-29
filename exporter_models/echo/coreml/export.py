"""Echo -> ExecuTorch Core ML (multi_pte, iOS 18, fp16 compute).

  echo_coreml_fp16.pte            backbone: encode_prefix + denoise, dynamic P/T, fp32 I/O
  echo_coreml_fp16_ane_t{T}.pte   backbone, static bucket P = 320, T = 128 / 256, CPU_AND_NE
  codec_decoder_coreml_fp16.pte   decode: codes [8, 250] -> audio [240000], static T, CPU_AND_GPU

Core ML puts a model on the ANE only with static shapes: the dynamic backbone
runs on the GPU whatever the compute units. The ANE buckets are the fastest
Echo on A18 (iPhone 16, 10 s: 865 vs 1262 ms MLX fp16) but barely gain on A19,
where MLX fp16 wins (NOTES.md). The caller pads to the bucket (key_bias /
self_bias, see _echo/backbone.py) and falls back when a clip does not fit.
The codec decoder runs slowest on the ANE (iPhone 16: 1090 ms vs 207 on
CPU_AND_GPU).

The codec decoder is static (T = 250, 10 s): the caller pads the codes and
trims the audio. RMSNorm stays aten.rms_norm ("native"), which Core ML keeps
whole and accumulates safely; a decomposed x * x overflows fp16.
"""

import importlib.util
import re
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent.parent
_PKG = "echo_coreml_echo"


def _vendored():
    if _PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _PKG, _MODEL_DIR / "_echo" / "__init__.py", submodule_search_locations=[str(_MODEL_DIR / "_echo")]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_PKG] = mod
        spec.loader.exec_module(mod)
    return tuple(importlib.import_module(f"{_PKG}.{m}") for m in ("backbone", "codec", "hub"))


_CODEC_T = 250
_BUCKET_P = 320


def _bucket_t(name: str):
    m = re.search(r"_ane_t(\d+)\.pte$", name)
    return int(m.group(1)) if m else None


def _compute_unit(ct, name: str):
    if _bucket_t(name):
        return ct.ComputeUnit.CPU_AND_NE
    if name.startswith("codec_decoder_"):
        return ct.ComputeUnit.CPU_AND_GPU
    return ct.ComputeUnit.ALL


def _programs(variant: dict) -> dict:
    backbone, codec, hub = _vendored()
    name = variant["file"]
    if _bucket_t(name):
        model = backbone.load(hub.checkpoint("fp32"), rms_norm="native")
        return backbone.export_bucket(model, _BUCKET_P, _bucket_t(name))
    if name.startswith("echo_"):
        return backbone.export(backbone.load(hub.checkpoint("fp32"), rms_norm="native"))
    if name.startswith("codec_decoder_"):
        return {"decode": codec.export_decoder(codec.load_codec(hub.codec()), t_static=_CODEC_T)}
    raise ValueError(f"unknown Echo program for variant file '{name}'")


def export(config: dict, out_dir: str):
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in config["variants"]:
        specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=precision[variant["precision"]],
            compute_unit=_compute_unit(ct, variant["file"]),
        )
        with torch.no_grad():
            eps = _programs(variant)
            program = to_edge_transform_and_lower(
                eps,
                partitioner=[CoreMLPartitioner(compile_specs=specs)],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            ).to_executorch()
        out_path = out_dir / variant["file"]
        out_path.write_bytes(program.buffer)
        written.append(str(out_path))
        del eps, program
    return written
