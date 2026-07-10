import importlib.util
import sys
from pathlib import Path

import torch

from exporter.dynamic_dims import build_dynamic_dims_program

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# The tokenizer's post-processor prepends <|startoftext|> (id 1). The native
# runtime tokenizer emits raw tokens, so forward prepends it to place the CLS
# token (this model pools index 0) first. Inputs are fed at their exact length
# with an all-ones mask -- no padding -- matching the createTextEmbedder
# contract (see the xnnpack export for the parity verification).
_BOS = 1


def _load_lfm_retrieval_model():
    name = f"lfm_retrieval_model_{_MODEL_DIR.name}_mlx"
    spec = importlib.util.spec_from_file_location(
        name, _MODEL_DIR / "_lfm_retrieval_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _BosPrepend(torch.nn.Module):
    """Prepends the CLS/startoftext token so the runtime can feed raw, unpadded
    tokens, then delegates to the CLS-pool + L2-norm embedding wrapper."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, attention_mask):
        bos = torch.tensor([[_BOS]], dtype=input_ids.dtype)
        one = torch.tensor([[1]], dtype=attention_mask.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
        attention_mask = torch.cat([one, attention_mask], dim=1)
        return self.inner(input_ids, attention_mask)


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.extension.llm.export.quantize import quantize_model_

    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]

    inner, _tok = _load_lfm_retrieval_model().load_embedding(torch.bfloat16)
    model = _BosPrepend(inner).eval()

    if variant["quantized"]:
        quantize_model_(model, qlinear_config="4w", qlinear_group_size=64)

    inputs = (
        torch.zeros(1, max_tokens, dtype=torch.int64),
        torch.ones(1, max_tokens, dtype=torch.int64),
    )
    dim = torch.export.Dim("tokens", min=1, max=max_tokens)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    ep = torch.export.export(model, inputs, dynamic_shapes=dynamic, strict=True)

    # Advertise the dynamic sequence dimension to the runtime. The raw inputs are
    # [1, tokens]: batch static, tokens dynamic in [1, max_tokens]; forward adds
    # the CLS token internally.
    seq_bounds = [[1, 1, 1], [1, max_tokens, 1]]
    dims = build_dynamic_dims_program([seq_bounds, seq_bounds])

    et_program = to_edge_transform_and_lower(
        {"forward": ep, "get_dynamic_dims_forward": dims},
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et_program.buffer)
    return str(out_path)
