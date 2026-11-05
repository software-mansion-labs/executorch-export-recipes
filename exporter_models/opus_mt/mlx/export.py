"""Opus-MT on MLX (Apple GPU).

The decoder's self-attention cache goes through the delegate's own
``mlx::kv_cache_update`` + ``mlx::custom_sdpa`` (``_DecoderLayer.cache_ops``),
so a decode step is one delegate call. Compute is bf16; the quantized variant
takes every Linear, the tied output projection included, to 8-bit weights.

The source length stays dynamic (1..256), as on XNNPACK.

Needs the software-mansion-labs executorch fork (the runtime-locked MLX
schema); ``require_mlx_runtime_fork`` refuses any other env.
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
    key = f"opus_mt_mlx{name}"
    spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from exporter.backends import require_mlx_runtime_fork

    require_mlx_runtime_fork()
    marian = _load("_marian")
    marian._DecoderLayer.cache_ops = "mlx"
    variant = config["variant"]
    model, cfg = marian.load(marian.source_repo(config["size"]), MAX_TARGET_LEN)
    model = model.to(torch.bfloat16)
    model.compute_dtype = torch.bfloat16
    if variant["quantized"]:
        from executorch.backends.mlx.llm.quantization import quantize_model_

        assert variant.get("quantize") == "8w", variant.get("quantize")
        quantize_model_(model, qlinear_config="8w")

    methods, schema = marian.export_methods(model, cfg, MAX_SOURCE_LEN, static=False)
    methods = {name: ep.run_decompositions({}) for name, ep in methods.items()}

    program = to_edge_transform_and_lower(
        methods,
        partitioner={name: [MLXPartitioner()] for name in methods},
        transform_passes=get_default_passes(),
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        constant_methods={"get_model_schema": marian.schema_json(schema)},
    ).to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=True))

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    _load("_tokenizer").write(marian.source_repo(config["size"]), out_path.parent / "tokenizer.json")
    return str(out_path)
