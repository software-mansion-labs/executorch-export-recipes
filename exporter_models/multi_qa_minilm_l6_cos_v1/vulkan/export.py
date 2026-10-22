from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import Dim, export as torch_export

from exporter.schema import load_and_validate_schema
from exporter.vulkan import embedding_off_texture, use_rank3_attention

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

_SOURCE = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
_CLS = 101
_SEP = 102


class _Embedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # eager attention, deliberately. The default SDPA path builds its
        # additive mask with -inf, and the Vulkan graph serializer writes
        # floats through flatc's JSON, which has no spelling for a non-finite
        # value ("cannot parse value starting with: -"). Eager builds the same
        # mask from finite values. It also partitions better: the whole model
        # lowers to one delegate blob.
        self.model = AutoModel.from_pretrained(
            _SOURCE, attn_implementation="eager"
        )
        # 14% on Adreno 840, 9% on Mali-G76, bit-identical on CPU.
        use_rank3_attention(self.model)

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
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]

    module = _Embedder().eval()
    inputs = (
        torch.zeros(1, max_tokens, dtype=torch.int64),
        torch.ones(1, max_tokens, dtype=torch.int64),
    )
    dim = Dim("tokens", min=1, max=max_tokens)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    ep = torch_export(module, inputs, dynamic_shapes=dynamic)
    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")

    # fp16: 2.05x XNNPACK on Adreno 840 against 1.30x for fp32, and half the
    # download. The 30522-row token embedding has to stay off texture storage
    # for fp16 to be correct at all -- see embedding_off_texture.
    with embedding_off_texture():
        program = to_edge_transform_and_lower(
            {"forward": ep},
            partitioner=[VulkanPartitioner(compile_options={"force_fp16": True})],
            constant_methods={"get_model_schema": schema_json},
        ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
