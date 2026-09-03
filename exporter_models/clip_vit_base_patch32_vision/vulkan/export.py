from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import export as torch_export

_SOURCE = "openai/clip-vit-base-patch32"


def _attention_rank3(self, hidden_states, attention_mask=None, **kwargs):
    """Multi-head attention with the singleton batch folded into the head dim.

    The stock CLIPAttention reshapes every projection to 4-D [B, H, T, D]. On
    Vulkan each of those reshapes is materialised as its own view dispatch: 216
    of them across the encoder, measured at 7.8% of GPU time. Batch is always 1
    here, so the identical math runs rank-3 and drops a third of them.
    """
    batch, tokens, channels = hidden_states.shape
    assert batch == 1, "rank-3 attention assumes a singleton batch"
    heads, head_dim = self.num_heads, self.head_dim

    def split(t):
        return t.view(tokens, heads, head_dim).permute(1, 0, 2)

    queries = split(self.q_proj(hidden_states)) * self.scale
    keys = split(self.k_proj(hidden_states))
    values = split(self.v_proj(hidden_states))

    scores = torch.bmm(queries, keys.transpose(1, 2))
    if attention_mask is not None:
        scores = scores + attention_mask.reshape(-1, tokens, tokens)
    weights = torch.softmax(scores, dim=-1)
    out = torch.bmm(weights, values).permute(1, 0, 2).reshape(1, tokens, channels)
    return self.out_proj(out), None


class _ClipVisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # eager attention, deliberately. transformers defaults to sdpa, whose
        # decomposition emits aten.mul.Scalar and aten.any.dim inside every
        # layer; Vulkan takes neither, so each layer splits the partition and
        # the encoder lowers to 25 delegate blobs instead of 1. The any.dim is
        # a masked-row guard that vision, which has no mask at all, never needs.
        model = AutoModel.from_pretrained(_SOURCE, attn_implementation="eager")
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection
        for layer in self.vision_model.encoder.layers:
            layer.self_attn.forward = _attention_rank3.__get__(layer.self_attn)

    def forward(self, pixel_values):
        pooled = self.vision_model(pixel_values).pooler_output
        return F.normalize(self.visual_projection(pooled), p=2, dim=1)


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    module = _ClipVisionEncoder().eval()
    inputs = (torch.zeros(shape, dtype=torch.float32),)
    ep = torch_export(module, inputs)

    program = to_edge_transform_and_lower(
        {"forward": ep},
        # fp16 is safe here: the vision tower has no token embedding, so it
        # never reaches the oversized-vocabulary texture path that breaks
        # fp16 embeddings (pytorch/executorch#22333). It also halves the
        # download, and fp32 on Vulkan measured slower than fp16 on Adreno.
        partitioner=[VulkanPartitioner(compile_options={"force_fp16": True})],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
