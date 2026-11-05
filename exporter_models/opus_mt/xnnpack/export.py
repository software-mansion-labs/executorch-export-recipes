"""Opus-MT on XNNPACK: encode and decode both fully on the CPU delegate.

The source length is dynamic (1..256), so a short sentence costs only its own
tokens. The decoder's self-attention KV cache is written with ``index_put``
on mutable buffers, which XNNPACK does not take; those two small copies per
block run on portable kernels between the delegated Linears. The reinplace pass
turns them into in-place ``index_put_``; without it each write was a
functional ``index_put`` plus a ``copy_`` back into the buffer, two full cache
copies per block per step and 16% of a decode step.

The int8 variant takes every Linear, the tied output projection included, to
8da8w with per-channel weights, and the input embedding table to int8
weight-only (``embedding_byte``). The projection is the largest single item
in a decode step: it streams the whole 512 x 58k matrix per token.
"""

import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256


def _load(name: str):
    key = f"opus_mt_xnnpack{name}"
    spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def _quantize(model, variant: dict):
    from torchao.quantization.granularity import PerAxis
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_

    assert variant.get("quantize") == "8da8w-pc", variant.get("quantize")
    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8, weight_granularity=PerAxis(0)),
        filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )
    if variant.get("embedding_quantize") == "int8":
        from executorch.examples.models.llama.source_transformation.quantize import (
            EmbeddingQuantHandler,
        )

        model = EmbeddingQuantHandler(
            model, bitwidth=8, group_size=None, packed=False, precision=torch.float32
        ).quantized_model()
    return model


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower

    marian = _load("_marian")
    marian._LayerNorm.decomposed = True
    variant = config["variant"]
    model, cfg = marian.load(marian.source_repo(config["size"]), MAX_TARGET_LEN)
    if variant["quantized"]:
        model = _quantize(model, variant)

    methods, schema = marian.export_methods(model, cfg, MAX_SOURCE_LEN, static=False)
    schema_json = marian.schema_json(schema)

    program = to_edge_transform_and_lower(
        methods,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_core_aten_ops_exception_list=[torch.ops.aten.linear.default]),
        transform_passes=[AddmmToLinearTransform()],
        constant_methods={"get_model_schema": schema_json},
    ).to_executorch(ExecutorchBackendConfig(run_reinplace_pass=True))

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    _load("_tokenizer").write(marian.source_repo(config["size"]), out_path.parent / "tokenizer.json")
    return str(out_path)
