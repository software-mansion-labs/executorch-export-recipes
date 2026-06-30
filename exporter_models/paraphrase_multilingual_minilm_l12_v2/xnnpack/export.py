from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import Dim, export as torch_export

from exporter.backends import lower

_SOURCE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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

    inputs = (
        torch.zeros(1, max_tokens, dtype=torch.int64),
        torch.ones(1, max_tokens, dtype=torch.int64),
    )
    dim = Dim("tokens", min=1, max=max_tokens)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    ep = torch_export(module, inputs, dynamic_shapes=dynamic)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
