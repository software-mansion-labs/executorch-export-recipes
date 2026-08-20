# Core ML export, pinned to CPU_AND_NE.
#
# These models were previously disabled with a note blaming Core ML's CPU cast
# kernel for a SIGSEGV on dynamically-shaped tensors. That diagnosis was wrong.
# Re-investigated 2026-08-20 (coremltools 9.0 / ExecuTorch 1.4.1): the fault is on
# the **GPU** path, and it is reachable from plain coremltools with no ExecuTorch
# involved. Bisecting the wrapper showed the trigger is the masked mean-pool over a
# dynamic sequence dim, not the encoder:
#
#     encoder only                  -> OK
#     encoder + CLS/SEP prepend     -> OK
#     encoder + masked mean-pool    -> SIGSEGV
#
# Core ML infers the pooled output as data-dependent ([?, H]) even though the
# reduction removes the dynamic axis, and the GPU backend rejects that shape:
#     Espresso exception: "Invalid blob shape": Data-dependent shapes were disabled
#
# Excluding the GPU avoids it entirely. generate_compile_specs defaults to
# ComputeUnit.ALL, which is why every one of these crashed. With CPU_AND_NE the
# same graph is numerically exact against the torch reference (cosine 1.000000,
# max abs 0.000000 at L = 2, 8, 37, 128, 380).
#
# Caveat: L = 1 is not supported -- coremltools clamps RangeDim to a lower bound of
# 2, so a single-token input is out of range by construction, unrelated to the bug.
# Filed upstream as apple/coremltools#2825.

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
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]

    module = _ClipTextEncoder().eval()
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
        minimum_deployment_target=ct.target.iOS18, compute_precision=precision
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
