from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import Dim, export as torch_export

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
    from executorch import exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig

    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]

    module = _ClipTextEncoder().eval()
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
