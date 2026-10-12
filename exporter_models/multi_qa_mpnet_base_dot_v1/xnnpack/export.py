from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import Dim, export as torch_export

from exporter.backends import lower
from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

_SOURCE = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
_CLS = 0
_SEP = 2


class _Embedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(_SOURCE)

    def _mean_pool(self, tokens, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
        return torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        cls = torch.tensor([[_CLS]], dtype=input_ids.dtype)
        sep = torch.tensor([[_SEP]], dtype=input_ids.dtype)
        input_ids = torch.cat([cls, input_ids, sep], dim=1)
        one = torch.tensor([[1]], dtype=attention_mask.dtype)
        attention_mask = torch.cat([one, attention_mask, one], dim=1)
        tokens = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled = self._mean_pool(tokens, attention_mask)
        return F.normalize(pooled, p=2, dim=1)


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]

    module = _Embedder().eval()

    if variant["quantized"]:
        import torch.nn as nn
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_

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

    inputs = (
        torch.zeros(1, max_tokens, dtype=torch.int64),
        torch.ones(1, max_tokens, dtype=torch.int64),
    )
    dim = Dim("tokens", min=1, max=max_tokens)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

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
