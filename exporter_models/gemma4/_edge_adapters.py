"""Adapters that fit the `_gemma4_edge` text recipes to the RNE text runner.

The vendored `_gemma4_edge` tree must stay byte-identical to gemma4_omni's, so
these patch names inside its `pipeline` module instead of editing it. Used by
the XNNPACK and Vulkan text exporters.
"""

import importlib.util
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent


def adapt(pipeline) -> None:
    """Applies both adapters below to an imported `_gemma4_edge` pipeline."""
    scalar_input_pos(pipeline)
    lean_loader(pipeline)


def lean_loader(pipeline):
    """Routes the pipeline's checkpoint load through `_lean_load.lean_load`.

    The edge port builds the model in fp32 and copies a full state dict into it
    before casting to `BASE_DTYPE`; that does not fit in 16 GB. Only the loader
    is wrapped, so quantisation and export run exactly as before.
    """
    spec = importlib.util.spec_from_file_location("gemma4_lean_load", _HERE / "_lean_load.py")
    lean = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lean)
    orig = pipeline.load_truncated_from_safetensors

    def load(*args, **kwargs):
        with lean.lean_load():
            return orig(*args, **kwargs)

    pipeline.load_truncated_from_safetensors = load


def scalar_input_pos(pipeline):
    """Adapts the text `forward` to the RNE text runner contract, in place.

    The edge port traces `forward(input_ids[1, S], input_pos[S]) -> [1, S|1,
    vocab]`; the runner calls `forward(tokens[1, S], pos[1]) -> [1, vocab]`. The
    vendored `_gemma4_edge` tree must stay byte-identical to gemma4_omni's, so
    instead of editing it this swaps two names inside `pipeline`: the forward
    wrapper (expand the start position, keep the last logits) and the trace call
    (a static [1] position instead of the seq-ranged one).
    """
    base = pipeline.SingleMethodForwardExport

    class ScalarPosForward(base):
        def forward(self, input_ids, input_pos):
            # input_pos[:1] is the start position whether the caller passes the
            # runner's [1] or the eager probe's [S] positions. The 0..S-1
            # offsets come from a cumsum over the tokens rather than
            # `arange(S)`: arange needs the symbolic seq length, and a Vulkan
            # partition whose only output is that SymInt fails memory planning
            # ("graph_output_allocated not set").
            offsets = torch.cumsum(torch.ones_like(input_ids[0]), dim=0) - 1
            positions = input_pos[:1] + offsets
            logits = super().forward(input_ids, positions)
            return logits[:, -1, :].to(torch.float32)

    orig_export = pipeline.torch_export

    def export_with_scalar_pos(module, args, *rest, dynamic_shapes=None, **kwargs):
        if isinstance(module, ScalarPosForward):
            input_ids, input_pos = args
            args = (input_ids, input_pos[:1].clone())
            dynamic_shapes = dict(dynamic_shapes, input_pos={})
        return orig_export(module, args, *rest, dynamic_shapes=dynamic_shapes, **kwargs)

    pipeline.SingleMethodForwardExport = ScalarPosForward
    pipeline.torch_export = export_with_scalar_pos
