# MLX: exports cleanly, but the artifact only computes at the declared MAX token
# length. Every shorter length fails at execute with error 0x1:
#
#     [MLXBackend.cpp:497] MLX execute failed: [broadcast_shapes]
#         Shapes (1,129,384) and (1,256,384) cannot be broadcast.
#
# The position-embedding buffer is specialized to the padded maximum
# (max_tokens + 2 for CLS/SEP), so any real input refuses to broadcast. Measured
# 2026-08-20 at L = max, max/2, 64, 32, 8, 2 and 1: only L = max succeeds.
#
# This is NOT a toolchain artifact. It was first seen with the PyPI-wheel export
# and re-tested after rebuilding with the software-mansion-labs fork and running
# under the fork runtime -- identical result both ways. (Worth stating because the
# YOLO fp16 MLX failure DID turn out to be a PyPI-runtime-only artifact, so the
# retest was not redundant.)
#
# Why that makes it unusable rather than merely limited: RNE's createTextEmbedder
# feeds tokens at their exact length with an all-ones mask and does NOT pad
# (len = min(ids.length, seqLen.max)), so in practice the model would throw on
# essentially every query.
#
# NOT universal to this family: distiluse_base_multilingual_cased_v2 uses the same
# dynamic Dim("tokens", 1..max) pattern and is correct at EVERY length (19 lengths
# from 1 to 126, cosine 1.000000, max abs 0.000000) and IS published. It loads
# through its own _distiluse_model.py rather than a bare AutoModel, which is the
# lead worth pulling if anyone revisits this.
#
# "mlx" is therefore removed from this model's model.json backends so batch
# exports skip it. The exporter is kept so the finding can be re-tested.

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import Dim, export as torch_export

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

_SOURCE = "sentence-transformers/all-MiniLM-L6-v2"
_CLS = 101
_SEP = 102


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
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    ep = torch_export(module, inputs, dynamic_shapes=dynamic)

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
