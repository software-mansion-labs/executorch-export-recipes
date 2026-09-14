"""LFM2.5-Embedding-350M -> ExecuTorch Vulkan (fp16).

A dense bi-encoder: 16 decoder layers, 10 of them short-conv (Conv1d) and 6
attention, CLS-pooled and L2-normalized to [1, 1024]. Conv1d only, so this is
exempt from the Adreno conv2d nondeterminism (pytorch/executorch#21938).

Unlike the BERT-family embedders in this backend, `use_rank3_attention` does
NOT apply here: it patches BertSelfAttention/MPNetSelfAttention/
DistilBertSelfAttention and raises when it patches nothing. Lfm2Attention is
its own class, so the rank-3 rewrite (9-14% elsewhere) is not available.
"""

import contextlib
import importlib.util
import sys
from pathlib import Path

import torch

from exporter.schema import load_and_validate_schema
from exporter.vulkan import embedding_off_texture, quantize_weight_only_int8

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# The tokenizer's post-processor prepends <|startoftext|> (id 1). The native
# runtime tokenizer emits raw tokens, so forward prepends it to place the CLS
# token (this model pools index 0) first. Mirrors xnnpack/export.py.
_BOS = 1


def _load_lfm_retrieval_model():
    name = f"lfm_retrieval_model_{_MODEL_DIR.name}_vulkan"
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


@contextlib.contextmanager
def _size_oblivious_export():
    """Treat backed sizes size-obliviously for the re-export after PT2E.

    The quantized graph re-export otherwise fails on
    `sym_min(64 + 64*T, 128 + 128*T) == 64 + 64*T` -- true for every T >= 0,
    from the short-conv's static-length padding, but torch.export will not
    prove it.
    """
    import torch.fx.experimental._config as fx_config

    prev = fx_config.backed_size_oblivious
    fx_config.backed_size_oblivious = True
    try:
        yield
    finally:
        fx_config.backed_size_oblivious = prev


def _skip_shortconv_linears(node) -> bool:
    """Keeps the short-conv projections out of int8.

    10 of the 16 decoder layers are `Lfm2ShortConv`, and quantizing their
    in/out projections scrambles the embedding -- `xnnpack/export.py` excludes
    the same `.conv.` linears from its 8da4w config for that reason. The module
    stack carries the fqn, so match on it the same way.
    """
    stack = node.meta.get("nn_module_stack") or {}
    return not any(".conv." in f"{fqn}." for fqn, _ in stack.values())


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]

    mod = _load_lfm_retrieval_model()
    inner, _tok = mod.load_embedding(torch.float32)
    # eager attention, deliberately. The default SDPA path builds its additive
    # mask with -inf, and the Vulkan graph serializer writes floats through
    # flatc's JSON, which has no spelling for a non-finite value ("cannot parse
    # value starting with: -"). Eager builds the same mask from finite values.
    inner.backbone.set_attn_implementation("eager")
    # No cache: the decoder layers otherwise emit 12 zero-element float32
    # placeholders (2 per attention layer), which the serializer parks in
    # `constant_buffer` rather than the constant segment. The Vulkan runtime
    # refuses that path outright:
    #   constant_buffer path is disabled (ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0)
    #   -> getTensorDataPtr() failed: 0x23
    # This is a bi-encoder scoring one full sequence, so the cache is dead weight.
    inner.backbone.config.use_cache = False
    # transformers >= 5.x calls the short-conv with a `seq_idx` kwarg the
    # model's remote non-causal override does not accept; re-patch with a
    # signature that tolerates it (numerics unchanged -- seq_idx is None here).
    mod.patch_shortconv_static_length(inner)
    model = _BosPrepend(inner).eval()

    inputs = (
        torch.zeros(1, max_tokens, dtype=torch.int64),
        torch.ones(1, max_tokens, dtype=torch.int64),
    )
    dim = torch.export.Dim("tokens", min=1, max=max_tokens)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    if variant["quantized"]:
        with _size_oblivious_export():
            model = quantize_weight_only_int8(
                model, inputs, dynamic, filter_fn=_skip_shortconv_linears
            )
            ep = torch.export.export(model, inputs, dynamic_shapes=dynamic, strict=True)
    else:
        ep = torch.export.export(model, inputs, dynamic_shapes=dynamic, strict=True)
    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")

    # The 65536-row token embedding is 4x the 16384 maxImageDimension2D bound,
    # so it has to stay off texture storage or it returns silently wrong values
    # under force_fp16 -- see embedding_off_texture.
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
