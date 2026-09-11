from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import Dim, export as torch_export
import torch.fx.experimental._config as fx_config

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

_SOURCE = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
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


# Size-oblivious, so the `min=1` on the token dim survives lowering. Without it
# `to_edge_transform_and_lower` widens the range to 2..max (torch.export itself
# keeps 1; the lift happens in to_edge), and the compiled Core ML model then
# rejects every single-word query with Error::DelegateInvalidHandle. That also
# put the artifact at odds with schema.json, which declares min 1. The decorator
# has to span the whole export, not just torch_export, because to_edge is where
# the widening happens; it restores the flag on return, so a batch export that
# runs several models in one process stays isolated.
# See react-native-executorch#1164.
@fx_config.patch(backed_size_oblivious=True)
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

    module = _Embedder().eval()
    inputs = (
        torch.zeros(1, max_tokens, dtype=torch.int64),
        torch.ones(1, max_tokens, dtype=torch.int64),
    )
    dim = Dim("tokens", min=1, max=max_tokens)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    ep = torch_export(module, inputs, dynamic_shapes=dynamic)
    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")

    # CPU_ONLY, deliberately. These graphs fail the Apple Neural Engine path
    # (coremltools#2825 masked mean-pool over a dynamic sequence dim, surfacing
    # as `Espresso exception: "Invalid blob shape": Data-dependent shapes were
    # disabled`; clip-text additionally `ANECCompile() FAILED`). Core ML still
    # returns correct output because it silently falls back to CPU, but the
    # failed ANE compile is paid on every call. Pinning CPU_ONLY skips the
    # attempt entirely. Measured on an iPhone 16, fp16, 3 warmup + 15 timed:
    #   all-MiniLM      13.7 ms -> 2.5 ms   5.5x
    #   multi-qa        14.1 ms -> 2.5 ms   5.6x
    #   paraphrase-L12  26.9 ms -> 4.2 ms   6.4x
    #   distiluse       33.0 ms -> 8.0 ms   4.1x
    #   clip-text       18.1 ms -> 4.9 ms   3.7x
    specs = CoreMLBackend.generate_compile_specs(
        compute_unit=ct.ComputeUnit.CPU_ONLY,
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
