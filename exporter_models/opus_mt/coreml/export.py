"""Opus-MT on Core ML.

``encode`` and ``decode`` get separate compute units, as whisper's do: the
encoder is a batched transformer stack that the Neural Engine can take, while
a decode step is one token of work whose cost is dispatch, so it runs
``CPU_ONLY``. The source length is static (``MAX_SOURCE_LEN``, padded and
masked by the caller), which is what lets the ANE compile the encoder.

The decoder's self-attention KV cache is handed to Core ML as model state
(``take_over_mutable_buffer``, iOS 18), so ``decode`` lowers to a single
delegate call instead of splitting around a portable ``index_put`` per block.
``decode`` is always fp32: Core ML's fp16 state corrupts the cache from the
second step (reproduces in plain coremltools), so ``precision`` only applies to
``encode``.
"""

import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 256


def _load(name: str):
    key = f"opus_mt_coreml{name}"
    spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    marian = _load("_marian")
    variant = config["variant"]
    # Per-method compute units and the decode precision; fp16 decode needs the
    # slice_scatter cache write (see _marian._DecoderLayer.cache_ops).
    units = {m: getattr(ct.ComputeUnit, variant.get(f"{m}_compute_unit", d))
             for m, d in (("encode", "ALL"), ("decode", "CPU_ONLY"))}
    decode_precision = variant.get("decode_precision", "fp32")
    if decode_precision == "fp16":
        marian._DecoderLayer.cache_ops = "slice_scatter"
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[variant["precision"]]
    model, cfg = marian.load(marian.source_repo(config["size"]), MAX_TARGET_LEN)

    methods, schema = marian.export_methods(model, cfg, MAX_SOURCE_LEN, static=True)
    schema_json = marian.schema_json(schema)

    def partitioner(compute_unit, compute_precision):
        return CoreMLPartitioner(
            compile_specs=CoreMLBackend.generate_compile_specs(
                compute_unit=compute_unit,
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=compute_precision,
            ),
            take_over_mutable_buffer=True,
        )

    program = to_edge_transform_and_lower(
        methods,
        partitioner={
            "encode": [partitioner(units["encode"], precision)],
            "decode": [partitioner(units["decode"], {"fp16": ct.precision.FLOAT16,
                                                      "fp32": ct.precision.FLOAT32}[decode_precision])],
        },
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        constant_methods={"get_model_schema": schema_json},
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    _load("_tokenizer").write(marian.source_repo(config["size"]), out_path.parent / "tokenizer.json")
    return str(out_path)
