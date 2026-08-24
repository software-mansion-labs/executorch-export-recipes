import importlib.util
import sys
from pathlib import Path

import torch

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# Mirrors the MLX/XNNPACK exports: the runtime tokenizer emits raw tokens, so
# forward prepends <|startoftext|> (id 1) to place the CLS token this model
# pools at index 0. Fed at exact length with an all-ones mask -- no padding.
_BOS = 1


def _load_lfm_retrieval_model():
    name = f"lfm_retrieval_model_{_MODEL_DIR.name}_coreml"
    spec = importlib.util.spec_from_file_location(
        name, _MODEL_DIR / "_lfm_retrieval_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _BosPrepend(torch.nn.Module):
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
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]

    mod = _load_lfm_retrieval_model()
    # float32 rather than the bfloat16 the MLX path uses -- Core ML has no bf16
    # input dtype and coremltools converts the graph from an fp32 trace.
    inner, _tok = mod.load_embedding(torch.float32)
    mod.patch_shortconv_static_length(inner)
    model = _BosPrepend(inner).eval()

    inputs = (
        torch.zeros(1, max_tokens, dtype=torch.int64),
        torch.ones(1, max_tokens, dtype=torch.int64),
    )
    dim = torch.export.Dim("tokens", min=1, max=max_tokens)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}
    ep = torch.export.export(model, inputs, dynamic_shapes=dynamic, strict=True)

    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")

    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )
    program = to_edge_transform_and_lower(
        {"forward": ep},
        partitioner=[CoreMLPartitioner(compile_specs=specs)],
        constant_methods={"get_model_schema": schema_json},
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
