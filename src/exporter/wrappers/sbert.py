import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..spec import Variant
from . import register


class SbertEmbedder(torch.nn.Module):
    def __init__(self, source: str, pooling: str, normalize: bool):
        super().__init__()
        self.model = AutoModel.from_pretrained(source)
        tok = AutoTokenizer.from_pretrained(source)
        self.cls_id = int(tok.cls_token_id)
        self.sep_id = int(tok.sep_token_id)
        self.pooling = pooling
        self.normalize = normalize

    def _pool(self, tokens, attention_mask):
        if self.pooling == "cls":
            return tokens[:, 0]
        mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
        return torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        cls = torch.tensor([[self.cls_id]], dtype=input_ids.dtype)
        sep = torch.tensor([[self.sep_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([cls, input_ids, sep], dim=1)
        one = torch.tensor([[1]], dtype=attention_mask.dtype)
        attention_mask = torch.cat([one, attention_mask, one], dim=1)

        tokens = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled = self._pool(tokens, attention_mask)
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled


@register("sbert")
def build(source: str, variant: Variant, *, pooling: str, normalize: bool) -> torch.nn.Module:
    return SbertEmbedder(source, pooling=pooling, normalize=normalize)
