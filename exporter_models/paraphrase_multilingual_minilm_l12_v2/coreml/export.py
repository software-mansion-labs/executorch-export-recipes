from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import Dim, export as torch_export

_SOURCE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_CLS = 0
_SEP = 2
# forward prepends CLS + appends SEP (+2), so the raw input_ids seq can be at
# most 510 for the post-concat sequence to stay within the model's 512 limit.
_MAX_TOKENS = 510


class _Embedder(torch.nn.Module):
    """Single-input embedder: input_ids only. The CoreML model is fed exactly
    the real tokens (no padding), so the attention mask is implicitly all-ones
    and derived internally rather than taken as a second input."""

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(_SOURCE)

    def _mean_pool(self, tokens, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
        return torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def forward(self, input_ids):
        cls = torch.tensor([[_CLS]], dtype=input_ids.dtype)
        sep = torch.tensor([[_SEP]], dtype=input_ids.dtype)
        input_ids = torch.cat([cls, input_ids, sep], dim=1)
        attention_mask = torch.ones_like(input_ids)
        tokens = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled = self._mean_pool(tokens, attention_mask)
        return F.normalize(pooled, p=2, dim=1)


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]

    module = _Embedder().eval()
    inputs = (torch.zeros(1, 8, dtype=torch.int64),)
    dim = Dim("tokens", min=1, max=_MAX_TOKENS)
    dynamic = {"input_ids": {1: dim}}

    ep = torch_export(module, inputs, dynamic_shapes=dynamic)
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )
    program = to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
