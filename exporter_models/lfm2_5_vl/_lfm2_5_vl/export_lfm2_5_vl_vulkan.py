# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export LFM2.5-VL as a single multi-method PTE delegated to the Vulkan backend
(Android GPU). Vulkan variant of export_lfm2_5_vl.py.

Differences from the XNNPACK export:
  - text_decoder + token_embedding lower with VulkanPartitioner; the
    vision_encoder stays on XNNPACK (see "Why the vision tower stays on
    XNNPACK" below).
  - the decoder's dynamic seq dim is capped at PREFILL_SEQ_CAP rather than the
    XNNPACK recipe's 2048, to keep per-chunk activations (which the Vulkan
    delegate holds in GPU memory) small; the runner chunks prefill to match.
  - the token embedding is int4 groupwise, not int8: Vulkan fuses 4-bit
    groupwise embeddings (`et_vk::embedding_q4gsw`) and nothing else, so int8
    would leave the whole method on host kernels.

Methods (D = text hidden dim: 2048 for 1.6B, 1024 for 450M):
  vision_encoder  : [1, 3, 512, 512] f32 NCHW pixels [0,255] -> [1, 256, D] f32
  token_embedding : [1, seq_len] i64                          -> [1, seq_len, D] f32
  text_decoder    : ([1, seq_len, D] f32, [seq_len] i64)      -> [1, 65536] f32

REQUIRES THE PATCHED VULKAN KERNELS, from:

    software-mansion-labs/executorch
    branch @ms/separate-backends          (ET 1.3.1 -- the same branch mlx uses)

The same prerequisite gemma4/vulkan carries: export and runtime must come from
one tree state. A PTE built against one set of shaders and run against another
produces gibberish at runtime, not an error, and that is the first thing to
check when on-device output is nonsense.

Usage:
    python export_lfm2_5_vl_vulkan.py \
        --model_dir LiquidAI/LFM2.5-VL-450M \
        [--quantize] [--output lfm2_5_vl_vulkan.pte]
"""

import contextlib
import logging
import os
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Optional

import torch

# Registers the `et_vk::` schemas the Vulkan partitioner matches against.
import executorch.backends.vulkan.custom_ops_lib  # noqa: F401
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.examples.models.llama.export_llama_lib import (
    get_quantizer_and_quant_params,
)
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    EmbeddingQuantHandler,
    get_quant_weight_transform,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from .model import (
    FIXED_H,
    FIXED_W,
    Lfm2p5VlModel,
    MAX_SEQ_LEN,
    IMAGE_SIZE,
    PATCH_SIZE,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from executorch.extension.llm.export.config.llm_config import LlmConfig
from torch.export import Dim
from torch.nn.attention import SDPBackend
from transformers.models.siglip2 import modeling_siglip2

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

# Upper bound on a single text_decoder prefill chunk, and therefore on the
# decoder's dynamic seq dim. The KV cache / context stays at MAX_SEQ_LEN.
#
# react-native-executorch's `multimodal_prefiller.cpp` reads `get_max_seq_len`
# as the decoder chunk size and bounds the WHOLE prefill against
# `get_max_context_len` when `enable_dynamic_shape` is true, so this cap costs
# more decoder calls per prompt, not correctness. (Upstream ExecuTorch's own
# multimodal prefiller does neither -- that is the hazard gemma4's unshipped
# `vulkan-mm` recipe exists to fix, and it does not apply to this runtime.)
#
# 256 matches the vision encoder's soft-token count, so an image's embeddings
# ride in a single chunk, and keeps the decoder's activation footprint an
# eighth of what the 2048-wide XNNPACK graph traces -- which matters more here,
# since the Vulkan delegate holds those activations in GPU memory.
PREFILL_SEQ_CAP = 256

# Group size for both the decoder's 8da4w linears and the int4 token embedding.
# 64, not the XNNPACK recipe's 128/32: Vulkan's `linear_dq8ca_q4gsw` /
# `embedding_q4gsw` are the 4-bit groupwise shaders, and 64 is the group size
# the device-validated gemma4/vulkan recipe uses.
LINEAR_GROUP_SIZE = 64
EMBEDDING_GROUP_SIZE = 64

# Whether Vulkan is allowed to claim the attention ops.
#
# True routes `llama::custom_sdpa` + `llama::update_cache` into the delegate,
# where Vulkan's `causal_sdpa` pattern fuses them into one
# `llama::sdpa_with_kv_cache` node per attention layer (this needs
# RemoveInversePermutePairs below to match at all). That gives a decoder with
# zero host ops -- and it is what FAILS ON DEVICE: SIGSEGV inside
# `Method::execute_instruction` on a 256-wide prefill chunk, and a thrown
# error on short chunks, at BOTH model sizes (Galaxy S26 Ultra, Adreno).
#
# False blocklists both ops instead, so LFM2.5's 6 attention layers run on
# host portable kernels (`sdpa_with_kv_cache_out` is in the shipped runtime)
# while the 10 short-conv layers, every linear, the RMS norms and RoPE stay on
# the GPU. Fragments the graph, but sidesteps the upstream Vulkan SDPA path
# entirely -- note that gemma4/vulkan, the only device-validated Vulkan LLM
# recipe here, also avoids it, via its own `et_vk::gemma_sdpa_with_kv_cache`
# kernels rather than upstream's.
DELEGATE_ATTENTION = True

# Whether Vulkan is allowed to claim LFM2.5's gated short-conv blocks.
#
# False keeps `aten.convolution` (the depthwise 1-D conv) and the conv-state
# `cat`/`slice_copy` on host. This exists because the decoder dies on device in
# Vulkan's dynamic-shape resize propagation:
#
#   vkcompute::vkapi::Error at Tensor.cpp:1393 virtual_resize
#   (new_sizes.size() == dim_order_.size())
#   "new sizes cannot modify the dimensionality of the tensor"
#
# i.e. some op's resize_fn hands a tensor a sizes list of a different RANK.
# The decoder cannot avoid dynamic shapes -- react-native-executorch prefills
# at S<=get_max_seq_len and then decodes at S=1 against the same method -- so
# resize propagation has to work, and for this graph it does not.
#
# The short conv is the prime suspect: it is what this architecture has and
# gemma4 (which does run on Vulkan) does not, and the failure is identical with
# attention fused and unfused, so attention is already ruled out.
DELEGATE_SHORTCONV = True


class RemoveInversePermutePairs(ExportPass):
    """Collapse `permute_copy(permute_copy(x, p), p_inv)` back to `x`.

    Not a micro-optimisation -- it is what makes the decoder's attention
    delegable at all.

    `CustomKVCache.update` returns `self.k_cache.transpose(1, 2)` (BSHD cache ->
    BHSD, llama's convention) and `SDPACustom.forward` immediately transposes
    back, because the custom op wants BSHD. The pair is an identity, but both
    survive into the edge graph, so `llama::custom_sdpa`'s key/value args are
    permutes rather than the cache placeholders themselves.

    Vulkan's `causal_sdpa` pattern (backends/vulkan/patterns/sdpa.py) refuses
    that shape: it reads `custom_sdpa.args[1]` and searches ITS users for the
    `llama::update_cache` node, so with a permute in between the search comes up
    empty and `replace_custom_sdpa_with_causal_sdpa` fails its assert -- the
    whole export dies inside `FusePatternsPass`, in the Vulkan backend, with no
    mention of attention. Vulkan's own `RemoveRedundantOpsTransform` does not
    help: it trims clones and dtype copies, not permutes, and it runs AFTER
    `FusePatternsPass` anyway.

    Running this at edge level, before partitioning, hands Vulkan the graph its
    matcher was written against: `custom_sdpa(q, k_cache, v_cache, ...)` with
    `update_cache` writing those same caches, which it then fuses into one
    `llama::sdpa_with_kv_cache` node per attention layer.
    """

    _PERMUTE = exir_ops.edge.aten.permute_copy.default

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        removed = 0
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target is not self._PERMUTE:
                continue
            inner = node.args[0]
            if (
                not isinstance(inner, torch.fx.Node)
                or inner.op != "call_function"
                or inner.target is not self._PERMUTE
            ):
                continue

            outer_dims, inner_dims = list(node.args[1]), list(inner.args[1])
            if len(outer_dims) != len(inner_dims):
                continue
            # Composition is identity iff inner_dims[outer_dims[i]] == i.
            if any(inner_dims[d] != i for i, d in enumerate(outer_dims)):
                continue

            node.replace_all_uses_with(inner.args[0])
            graph_module.graph.erase_node(node)
            removed += 1

        if removed:
            logging.info("RemoveInversePermutePairs: removed %d permute pairs", removed)
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, removed > 0)


def _resolve_params_path(model_dir: str, params: Optional[str]) -> Optional[str]:
    """Pick a bundled config based on model_dir if --params was not provided."""
    if params is not None:
        return params
    name = model_dir.lower()
    if "450m" in name:
        return os.path.join(_CONFIG_DIR, "lfm2_5_vl_450m_config.json")
    if "1.6b" in name or "1_6b" in name:
        return os.path.join(_CONFIG_DIR, "lfm2_5_vl_1_6b_config.json")
    return None


def _vulkan_partitioner():
    """One partitioner instance per method.

    `require_dynamic_shapes` is what gemma4/vulkan passes and what the decoder
    needs -- without it the partitioner refuses the symbolic seq dim outright.
    `torchao.dequantize_affine` is blocklisted so a dequant node that did NOT
    fuse into a quantised-linear pattern stays on the host instead of being
    claimed as a standalone GPU op.
    """
    blocklist = {exir_ops.edge.torchao.dequantize_affine.default}
    if not DELEGATE_SHORTCONV:
        blocklist |= {
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.slice_copy.Tensor,
        }
    if not DELEGATE_ATTENTION:
        # Keep attention out of the delegate. Blocking `custom_sdpa` also keeps
        # Vulkan's `causal_sdpa` matcher from ever seeing it, which matters:
        # that matcher asserts rather than declining when the graph does not
        # match its expected shape.
        # Edge-dialect overloads, not `torch.ops.*` -- the partitioner matches
        # against the edge graph's node targets.
        # Both halves are required, and neither works alone:
        #
        #  * `_without_causal_sdpa_pattern()` (see export_all) stops the
        #    pattern matcher claiming these nodes. Without it this blocklist is
        #    never even consulted -- VulkanPartitioner's `is_node_supported`
        #    returns True for any pattern-matched node BEFORE reaching it.
        #  * this blocklist stops the partitioner picking the same ops up
        #    individually afterwards, which it otherwise does: `llama::
        #    custom_sdpa` and `llama::update_cache` have their own op_registry
        #    entries, so dropping the pattern alone just yields UNFUSED Vulkan
        #    attention rather than host attention.
        blocklist |= {
            exir_ops.edge.llama.custom_sdpa.default,
            exir_ops.edge.llama.update_cache.default,
            exir_ops.edge.llama.sdpa_with_kv_cache.default,
            "llama::custom_sdpa",
            "llama::update_cache",
            "llama::sdpa_with_kv_cache",
        }

    return VulkanPartitioner(
        compile_options={"require_dynamic_shapes": True},
        operator_blocklist=blocklist,
    )


@contextlib.contextmanager
def _without_causal_sdpa_pattern():
    """Unregister Vulkan's `causal_sdpa` pattern for the duration of lowering.

    The only lever that actually keeps attention on the host. It has to be the
    pattern registry rather than `operator_blocklist` because the partitioner
    short-circuits to "supported" for any pattern-matched node before it looks
    at the blocklist (vulkan_partitioner.py, `is_node_supported`).

    Dropping the entry disables both halves at once: `get_all_fusable_subgraphs`
    stops claiming the nodes for the delegate, and `replace_all_fusable_subgraphs`
    stops trying to rewrite them (which is also what fires the AssertionError
    this recipe would otherwise trip over).
    """
    from executorch.backends.vulkan.patterns.pattern_registry import (
        fusable_patterns,
    )

    entry = fusable_patterns.pop("causal_sdpa", None)
    if entry is None:
        logging.warning(
            "vulkan 'causal_sdpa' pattern not found -- attention exclusion may "
            "not have taken effect; check delegation in the produced .pte"
        )
    try:
        yield
    finally:
        if entry is not None:
            fusable_patterns["causal_sdpa"] = entry


class Lfm2p5VlEdgeManager(LLMEdgeManager):
    """strict=False + SDPBackend.MATH, like the XNNPACK export."""

    def export(self) -> "Lfm2p5VlEdgeManager":
        dynamic_shape = self._get_dynamic_shape()
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            self.export_program = torch.export.export(
                self.model,
                self.example_inputs,
                dynamic_shapes=dynamic_shape,
                strict=False,
            )
            self.pre_autograd_graph_module = self.export_program.module()
        return self


def _bmm_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    """`eager_attention_forward` with the batched matmuls written as 3-D bmm.

    Mathematically identical -- and bit-identical in practice -- but it avoids
    the `expand_copy` pair that edge dialect emits when lowering a 4-D matmul.
    See _unmasked_eager_vision_attention.
    """
    b, h, s, d = query.shape
    kv_s = key.shape[2]
    q3 = query.reshape(b * h, s, d)
    k3 = key.reshape(b * h, kv_s, d).transpose(1, 2)
    v3 = value.reshape(b * h, kv_s, d)

    attn = torch.bmm(q3, k3) * scaling
    if attention_mask is not None:
        attn = attn + attention_mask.reshape(b * h, s, kv_s)
    attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.bmm(attn, v3)
    return out.reshape(b, h, s, d).transpose(1, 2).contiguous(), None


@contextlib.contextmanager
def _unmasked_eager_vision_attention(lfm2):
    """Drop the vision tower's attention mask and its SDPA guard for export.

    Every patch of a 512x512 image is valid -- `encode_images` passes an
    all-ones `pixel_attention_mask` -- so the encoder's mask is uniformly
    permissive and computing it is pure overhead. Two separate costs follow
    from it, and removing one without the other barely helps:

      * building the mask, and
      * SDPA's fully-masked-row NaN guard (`eq`/`any`/`logical_not`/
        `full_like`/`mul` per layer), which torch's decomposition emits
        whether or not a mask is actually present.

    Calling the encoder with `attention_mask=None` kills the first;
    `_attn_implementation="eager"` kills the second, since
    `eager_attention_forward` is a plain matmul/softmax/matmul that the
    partitioner can claim whole.

    Eager attention alone still leaves 48 host ops, because edge dialect lowers
    a 4-D `matmul` to `expand_copy` + `bmm`, and XNNPACK does not claim
    `expand_copy`. Both operands already have the output shape here, so each of
    those expands is a pure ~3 MB copy that computes nothing -- 2 matmuls x 2
    operands x 12 layers. `_bmm_attention_forward` below folds the head dim into
    the batch dim and calls `bmm` directly, so there is nothing to broadcast.

    Together, on the 450M tower:

        as shipped         51 delegate calls, 229 host ops
        unmasked eager     51,                 75
        + bmm attention    27,                 27

    What is left is 25 `native_layer_norm` and 2 `unfold_copy`. XNNPACK has no
    layer-norm partitioner config at all, so those 25 cannot be delegated to it.

    Numerics: dropping the mask is bit-identical, and so is the bmm rewrite.
    Eager attention reassociates the reduction, which moves the tower output by
    ~1.5e-6 mean absolute against a signal of magnitude ~0.19 (7.6e-6 relative)
    -- orders of magnitude below the 8da4w quantization error this export
    already accepts.
    """
    tower = lfm2.model_.model.vision_tower
    modules = [m for m in tower.modules() if hasattr(m, "config")]
    saved = [(m, m.config._attn_implementation) for m in modules]
    saved.append((tower, tower.config._attn_implementation))
    for module, _ in saved:
        module.config._attn_implementation = "eager"
    saved_attn = modeling_siglip2.eager_attention_forward
    modeling_siglip2.eager_attention_forward = _bmm_attention_forward
    try:
        yield tower
    finally:
        modeling_siglip2.eager_attention_forward = saved_attn
        for module, impl in saved:
            module.config._attn_implementation = impl


def export_image_encoder(lfm2) -> torch.export.ExportedProgram:
    """Export vision encoder as 'vision_encoder' method (fp32, XNNPACK).

    Input:  [1, 3, 512, 512] float32 NCHW pixels in [0, 255]
    Output: [1, 256, D]      float32 image embeddings
    """

    class ImageEncoder(torch.nn.Module):
        def __init__(self, lfm2, tower):
            super().__init__()
            self.lfm2 = lfm2
            self.tower = tower

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # Inlined encode_images so the tower can be called without a mask;
            # see _unmasked_eager_vision_attention for why.
            x = images / 255.0
            x = (x - 0.5) / 0.5
            x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
            x = x.permute(0, 2, 3, 4, 5, 1).reshape(
                1, FIXED_H * FIXED_W, PATCH_SIZE * PATCH_SIZE * 3
            )
            spatial_shapes = torch.tensor([[FIXED_H, FIXED_W]], dtype=torch.int64)
            hidden = self.tower.embeddings(x, spatial_shapes)
            encoded = self.tower.encoder(inputs_embeds=hidden, attention_mask=None)
            feats = self.tower.post_layernorm(encoded.last_hidden_state)
            feats = feats.reshape(feats.shape[0], FIXED_H, FIXED_W, -1)
            projector = self.lfm2.model_.model.multi_modal_projector
            feats = feats.to(projector.linear_1.weight.dtype)
            projected = projector(feats)
            return projected.reshape(1, -1, projected.shape[-1])

    example_pixels = torch.randint(
        0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32
    )
    logging.info("Exporting vision encoder (fp32, unmasked eager attention)...")
    with _unmasked_eager_vision_attention(lfm2) as tower, torch.no_grad():
        return torch.export.export(
            ImageEncoder(lfm2, tower), (example_pixels,), strict=False
        )


def export_text_decoder(
    lfm2,
    quantize: bool,
    dtype: DType = DType.fp32,
) -> torch.export.ExportedProgram:
    """Export hybrid LFM2.5 decoder as 'text_decoder' method.

    Same source transforms as the XNNPACK recipe: the Vulkan op registry
    carries `llama::sdpa_with_kv_cache`, so the ET custom SDPA + custom KV
    cache are delegable here rather than something to route around.
    """
    class TextDecoder(torch.nn.Module):
        def __init__(self, text_model):
            super().__init__()
            self.text_model = text_model

        def forward(
            self, embeddings: torch.Tensor, input_pos: torch.Tensor
        ) -> torch.Tensor:
            return self.text_model(None, {"input_pos": input_pos}, embeddings)

    decoder = TextDecoder(lfm2.text_model)
    dim = lfm2.text_model_args.dim
    dummy_seq = 8
    dummy_embeddings = torch.randn(1, dummy_seq, dim, dtype=dtype.to_torch_dtype())
    dummy_input_pos = torch.arange(dummy_seq, dtype=torch.int64)
    token_dim = Dim("token_dim", min=1, max=PREFILL_SEQ_CAP)
    dynamic_shapes = ({1: token_dim}, {0: token_dim})

    manager = Lfm2p5VlEdgeManager(
        model=decoder,
        modelname="lfm2_5_vl_text_decoder",
        max_seq_len=PREFILL_SEQ_CAP,
        dtype=dtype,
        use_kv_cache=True,
        example_inputs=(dummy_embeddings, dummy_input_pos),
        dynamic_shapes=dynamic_shapes,
    )

    source_transforms = [
        replace_kv_cache_with_custom_kv_cache,
        replace_sdpa_with_custom_op,
    ]

    if quantize:
        llm_config = LlmConfig()
        llm_config.quantization.qmode = "8da4w"
        llm_config.quantization.group_size = LINEAR_GROUP_SIZE
        quant_transform = get_quant_weight_transform(
            quantization_mode=llm_config.quantization.qmode,
            group_size=llm_config.quantization.group_size,
            computation_dtype=dtype,
            checkpoint_path=None,
            tokenizer_path=None,
            calibration_tasks=None,
            calibration_limit=None,
            calibration_seq_length=None,
        )
        _, quantizers, _ = get_quantizer_and_quant_params(llm_config)
        source_transforms.append(quant_transform)
        logging.info(
            "Exporting text decoder (8da4w quantized, group %d)...",
            LINEAR_GROUP_SIZE,
        )
        manager = (
            manager.set_output_dir("./")
            .to_dtype(dtype)
            .source_transform(source_transforms)
            .export()
            .pt2e_quantize(quantizers)
        )
    else:
        logging.info("Exporting text decoder (fp32)...")
        manager = (
            manager.set_output_dir("./")
            .to_dtype(dtype)
            .source_transform(source_transforms)
            .export()
        )

    return manager.export_program


def export_token_embedding(lfm2, quantize: bool) -> torch.export.ExportedProgram:
    """Export the token embedding table as the 'token_embedding' method.

    int4 groupwise, where the XNNPACK recipe uses int8/g32. Not a preference:
    Vulkan's embedding fusion (backends/vulkan/patterns/quantized_embedding.py)
    matches `quantized_decomposed.embedding_4bit.dtype` with quant_min/max
    -8/7 and no zero-points, and nothing else. int8 emits `embedding_byte`,
    which no Vulkan pattern claims, so the whole method would fall back to
    host kernels. int4/g64 is also what the device-validated gemma4/vulkan
    recipe uses for its main embeddings.

    This is the deviation from the XNNPACK artifact most likely to show up as
    a quality difference, so it is the first thing to bisect if generation
    degrades against the XNNPACK build.

    IMPORTANT: call this AFTER export_text_decoder. EmbeddingQuantHandler
    mutates the module in place.
    """
    language_model = lfm2.model_.model.language_model
    token_dim = Dim("token_dim_1", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = [{1: token_dim}]

    if quantize:
        logging.info("Exporting token embedding (int4 groupwise, packed)...")
        quantized_lm = EmbeddingQuantHandler(
            language_model,
            bitwidth=4,
            group_size=EMBEDDING_GROUP_SIZE,
            packed=True,
        ).quantized_model()
        embed_module = quantized_lm.embed_tokens
    else:
        logging.info("Exporting token embedding (fp32)...")
        embed_module = language_model.get_input_embeddings()

    example_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64)
    with torch.no_grad():
        return torch.export.export(
            embed_module, (example_ids,), dynamic_shapes=dynamic_shapes, strict=False
        )


def export_all(
    model_dir: str,
    output: Optional[str],
    dtype: DType = DType.fp32,
    quantize: bool = False,
    max_seq_len: int = MAX_SEQ_LEN,
    max_context_len: int = MAX_SEQ_LEN,
    params_path: Optional[str] = None,
    _return_program: bool = False,
):
    logging.info(f"Loading {model_dir}...")
    lfm2_model = Lfm2p5VlModel(
        model_dir=model_dir,
        max_seq_len=max_seq_len,
        max_context_len=max_context_len,
        params_path=params_path,
    )
    lfm2 = lfm2_model.get_eager_model()
    if dtype != DType.fp32:
        lfm2 = lfm2.to(dtype.to_torch_dtype())

    logging.info("[1/3] Exporting vision encoder...")
    vision_ep = export_image_encoder(lfm2)

    logging.info("[2/3] Exporting text decoder...")
    decoder_ep = export_text_decoder(lfm2, quantize, dtype)

    logging.info("[3/3] Exporting token embedding...")
    token_ep = export_token_embedding(lfm2, quantize)

    logging.info("Lowering to Edge IR + Vulkan...")
    attention_ctx = (
        contextlib.nullcontext()
        if DELEGATE_ATTENTION
        else _without_causal_sdpa_pattern()
    )
    with attention_ctx:
        lowered = to_edge_transform_and_lower(
            {
                "vision_encoder": vision_ep,
                "token_embedding": token_ep,
                "text_decoder": decoder_ep,
            },
            # The permute-collapsing pass exists only to make Vulkan's SDPA
            # matcher fire; with attention on host it is pointless (and the
            # permutes are what the host kernels expect).
            transform_passes=(
                {"text_decoder": [RemoveInversePermutePairs()]}
                if DELEGATE_ATTENTION
                else None
            ),
            partitioner={
                # Why the vision tower stays on XNNPACK: it runs once per image,
                # so it is not where GPU decode speed is won, and the SigLIP2
                # tower is the part of this model most likely to fragment on
                # Vulkan. Keeping it on the validated XNNPACK path makes the
                # decoder the only variable being changed here.
                "vision_encoder": [XnnpackPartitioner()],
                "token_embedding": [_vulkan_partitioner()],
                "text_decoder": [_vulkan_partitioner()],
            },
            constant_methods={
                # get_max_seq_len is the runner's PER-CALL PREFILL CHUNK SIZE, not
                # the context limit -- get_max_context_len is. Both report what the
                # graph actually accepts: a runner that chunks at more than the
                # graph's bound gets silent garbage, not an error.
                "get_max_seq_len": PREFILL_SEQ_CAP,
                "get_max_context_len": lfm2.text_model_args.max_context_len,
                "get_n_layers": lfm2.text_model_args.n_layers,
                "get_vocab_size": lfm2.text_model_args.vocab_size,
                "use_kv_cache": lfm2.text_model_args.use_kv_cache,
                "use_sdpa_with_kv_cache": lfm2.text_model_args.use_sdpa_with_kv_cache_op,
                # True so the runner budgets total prefill against
                # get_max_context_len and chunks by get_max_seq_len, rather than
                # capping the whole prefill at one chunk. The graph is still
                # chunk-static; the model's own enable_dynamic_shape stays False
                # to avoid the .item() in rope.get_freqs that breaks export.
                "enable_dynamic_shape": True,
                # EOS = <|im_end|> (token 7).
                "get_eos_ids": [7],
            },
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

    logging.info("Finalizing ExecuTorch program...")
    et_program = lowered.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[QuantFusionPass()] if quantize else [],
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass={
                "vision_encoder": ConstraintBasedSymShapeEvalPass(),
                "token_embedding": ConstraintBasedSymShapeEvalPass(),
                "text_decoder": ConstraintBasedSymShapeEvalPass(),
            },
        )
    )

    for execution_plan in et_program._emitter_output.program.execution_plan:
        logging.info(
            f"Required memory for activation in bytes: {execution_plan.non_const_buffer_sizes}"
        )

    if _return_program:
        return et_program

    logging.info(f"Saving {output}...")
    with open(output, "wb") as f:  # type: ignore[arg-type]
        et_program.write_to_file(f)
    logging.info(f"Saved {output}. Methods: {et_program.methods}")


def main():
    parser = ArgumentParser(description="Export LFM2.5-VL to ExecuTorch (Vulkan)")
    parser.add_argument("--model_dir", default="LiquidAI/LFM2.5-VL-450M")
    # fp32 only. gemma4/vulkan runs its stack at fp16 via a bespoke retarget
    # (bf16 weights -> fp16, scales kept fp32); nothing here has been measured
    # at fp16, so the knob is not offered rather than offered untested.
    parser.add_argument("--dtype", default="fp32", choices=["fp32"])
    parser.add_argument(
        "--quantize",
        default=False,
        action=BooleanOptionalAction,
        help="8da4w (group 64) quantize the decoder linears",
    )
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max_context_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--params", default=None)
    parser.add_argument("--output", default="lfm2_5_vl_vulkan.pte")
    args = parser.parse_args()

    export_all(
        model_dir=args.model_dir,
        output=args.output,
        dtype=DType.fp16 if args.dtype == "fp16" else DType.fp32,
        quantize=args.quantize,
        max_seq_len=args.max_seq_len,
        max_context_len=args.max_context_len,
        params_path=_resolve_params_path(args.model_dir, args.params),
    )


if __name__ == "__main__":
    main()
