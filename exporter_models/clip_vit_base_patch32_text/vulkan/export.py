from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import Dim, export as torch_export

from exporter.schema import load_and_validate_schema
from exporter.vulkan import embedding_off_texture

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

_SOURCE = "openai/clip-vit-base-patch32"
_BOS = 49406
_EOS = 49407


def _attention_rank3(self, hidden_states, attention_mask=None, **kwargs):
    """Multi-head attention with the singleton batch folded into the head dim.

    See the note in the vision export: the stock 4-D reshapes each become their
    own Vulkan view dispatch, and batch is always 1 here.
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


class _ClipTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # eager attention: see the note in the vision export. sdpa fragments the
        # encoder into 26 delegate blobs; eager lowers it to 3.
        model = AutoModel.from_pretrained(_SOURCE, attn_implementation="eager")
        self.text_model = model.text_model
        self.text_projection = model.text_projection
        for layer in self.text_model.encoder.layers:
            layer.self_attn.forward = _attention_rank3.__get__(layer.self_attn)

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
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]

    module = _ClipTextEncoder().eval()
    inputs = (
        torch.zeros(1, max_tokens, dtype=torch.int64),
        torch.ones(1, max_tokens, dtype=torch.int64),
    )
    dim = Dim("tokens", min=1, max=max_tokens)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    ep = torch_export(module, inputs, dynamic_shapes=dynamic)
    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")

    # fp16 halves the download and is the faster arm on Adreno, but the
    # 49408-row token embedding has to stay off texture storage for it to be
    # correct at all -- see embedding_off_texture.
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
