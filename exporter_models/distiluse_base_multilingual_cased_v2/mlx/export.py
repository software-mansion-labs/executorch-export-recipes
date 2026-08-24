import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import Dim, export as torch_export

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_MAX_TOKENS = 126  # model max_seq_length 128 - 2 for CLS/SEP


def _load_model():
    name = "distiluse_mlx_model"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_distiluse_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch import exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig

    variant = config["variant"]

    module = _load_model().load_embedder()

    if variant.get("quantized"):
        # 68% of this model's parameters are in the token embedding (large
        # multilingual vocab), so quantizing Linear alone only reaches 412 MB.
        # The embedding CAN be quantized here: unlike Whisper, this is an
        # encoder with no tied output projection, so the embedding is a pure
        # lookup and never serves a raw matmul.
        #
        # Measured min cosine vs the fp32 reference over multilingual probes:
        #   int8 linear      412 MB  0.999859
        #   int8 linear+emb  136 MB  0.999790   <- shipped
        #   int4 linear+emb  146 MB  0.979790   (visibly worse, not offered)
        from executorch.backends.mlx.llm.quantization import quantize_model_

        quantize_model_(module, qlinear_config="8w", qembedding_config="8w")

    inputs = (
        torch.zeros(1, 8, dtype=torch.int64),
        torch.ones(1, 8, dtype=torch.int64),
    )
    dim = Dim("tokens", min=1, max=_MAX_TOKENS)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    ep = torch_export(module, inputs, dynamic_shapes=dynamic)

    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")

    program = exir.to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        constant_methods={"get_model_schema": schema_json},
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
