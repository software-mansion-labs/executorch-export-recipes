# MLX: exports, but the artifact is unusable at the token lengths RNE feeds.
#
# Measured 2026-08-20, fork toolchain, fork runtime, Dim("tokens", 1..382):
#     EVERY length FAILS with error 0x1, including the declared max 382 --
#     strictly worse than all_mpnet_base_v2, which at least works at max length
#
# Same family failure as all_minilm_l6_v2 / multi_qa_minilm_l6_cos_v1 /
# paraphrase_multilingual_minilm_l12_v2 / clip_vit_base_patch32_text -- the
# position-embedding buffer is specialized to the padded maximum, so shorter
# inputs refuse to broadcast. RNE's createTextEmbedder feeds tokens at their exact
# length with no padding, so this throws on essentially every real query.
#
# Not a toolchain artifact: re-tested after rebuilding with the software-mansion-labs
# fork and running under the fork runtime, identical result.
#
# NOT universal: distiluse_base_multilingual_cased_v2 uses the same dynamic-Dim
# pattern and is correct at every length (cosine 1.000000), and IS published. It
# loads through its own _distiluse_model.py rather than a bare AutoModel.
#
# "mlx" is removed from this model's model.json backends. The exporter is kept so
# the finding can be re-tested.

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import Dim, export as torch_export

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
    from executorch import exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig

    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]

    module = _Embedder().eval()
    inputs = (
        torch.zeros(1, max_tokens, dtype=torch.int64),
        torch.ones(1, max_tokens, dtype=torch.int64),
    )
    dim = Dim("tokens", min=1, max=max_tokens)
    ep = torch_export(module, inputs, dynamic_shapes=({1: dim}, {1: dim}))

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
