import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import Dim, export as torch_export

from exporter.backends import lower
from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_MAX_TOKENS = 126  # model max_seq_length 128 - 2 for CLS/SEP


def _load_model():
    name = "distiluse_xnnpack_model"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_distiluse_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import torch.nn as nn
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_

    variant = config["variant"]
    module = _load_model().load_embedder()

    inputs = (
        torch.zeros(1, 8, dtype=torch.int64),
        torch.ones(1, 8, dtype=torch.int64),
    )
    dim = Dim("tokens", min=1, max=_MAX_TOKENS)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    if variant["quantized"]:
        qcfg = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4, weight_granularity=PerGroup(32)
        )
        quantize_(module, qcfg, filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear))

    # 8da4w covers the Linears only, which left the fp32 word-embedding table as
    # 82-97% of the file. Quantize it weight-only to embedding_byte/embedding_4bit
    # (portable kernels, both linked into the RNE runtime).
    embedding_quantize = variant.get("embedding_quantize")
    if embedding_quantize:
        from executorch.examples.models.llama.source_transformation.quantize import (
            EmbeddingQuantHandler,
        )

        bits = {"int8": 8, "int4": 4}[embedding_quantize]
        module = EmbeddingQuantHandler(
            module,
            bitwidth=bits,
            group_size=None,
            packed=bits == 4,
            precision=torch.float32,
        ).quantized_model()

    ep = torch_export(module, inputs, dynamic_shapes=dynamic)

    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")
    program = lower(
        {"forward": ep},
        config["backend"],
        constant_methods={"get_model_schema": schema_json},
    )

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
