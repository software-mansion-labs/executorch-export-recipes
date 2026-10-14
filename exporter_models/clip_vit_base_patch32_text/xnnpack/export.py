from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import Dim, export as torch_export

from exporter.backends import lower
from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

_SOURCE = "openai/clip-vit-base-patch32"
_BOS = 49406
_EOS = 49407


class _ClipTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModel.from_pretrained(_SOURCE)
        self.text_model = model.text_model
        self.text_projection = model.text_projection

    def forward(self, input_ids, attention_mask):
        bos = torch.tensor([[_BOS]], dtype=input_ids.dtype)
        eos = torch.tensor([[_EOS]], dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids, eos], dim=1)
        one = torch.tensor([[1]], dtype=attention_mask.dtype)
        attention_mask = torch.cat([one, attention_mask, one], dim=1)

        pooled = self.text_model(input_ids, attention_mask).pooler_output
        projected = self.text_projection(pooled)
        return F.normalize(projected, p=2, dim=1)


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]

    module = _ClipTextEncoder().eval()
    inputs = (
        torch.zeros(1, max_tokens, dtype=torch.int64),
        torch.ones(1, max_tokens, dtype=torch.int64),
    )
    if variant["quantized"]:
        # 8da4w costs 2.2 points of ImageNetV2 zero-shot top-1 here (min text cosine
        # 0.74); int8 per-channel weights keep the tower close to fp32.
        import torch.nn as nn
        from torchao.quantization.granularity import PerAxis
        from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_

        assert variant.get("quantize") == "8da8w-pc", variant.get("quantize")
        qcfg = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int8, weight_granularity=PerAxis(0)
        )
        quantize_(module, qcfg, filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear))

    # That covers the Linears only; the token embedding table (40% of the weights)
    # is quantized weight-only to embedding_byte/embedding_4bit (portable kernels,
    # both linked into the RNE runtime).
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
