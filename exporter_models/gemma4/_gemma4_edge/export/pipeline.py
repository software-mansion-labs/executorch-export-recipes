"""Build one Gemma4 PTE from a `Recipe`: load -> transform -> quantize ->
export -> lower -> write.

This is the distillation of four export drivers that were once maintained as
separate scripts:

    xnnpack-text   single `forward`
    xnnpack-mm     4 methods
    vulkan-text    single `forward`
    vulkan-mm      4 methods

Everything that differs between them is a field on the `Recipe`; everything
that is the same is written once here.

ORDERING IS THE WHOLE POINT OF THIS FILE
----------------------------------------
Nine constraints, every one of which has cost someone a full export cycle:

 1. Class-level flags on `Gemma4Attention` / `RMSNorm` are read during
    `__init__` and inside `forward`; they must be set BEFORE the model is
    constructed.
 2. `RMSNorm` is a SEPARATE class in `gemma4.model.text` and in
    `gemma4.model.audio`. Setting `use_rsqrt` on one silently misses the
    other.
 3. `split_ple_for_export()` runs AFTER `load_state_dict` (it slices the
    loaded PLE tensors) and BEFORE quantize (the quantizer matches on the
    `_ple_per_layer_*` FQNs that only exist after the split).
 4. quantize AFTER `split_ple_for_export`.
 5. `setup_caches` AFTER load.
 6. `replace_kv_cache_with_quantized_kv_cache` AFTER `setup_caches` (there
    are no caches to swap before it).
 7. `replace_sliding_kv_cache_with_ring_kv_cache` AFTER the int8 swap, so
    it picks up the `Gemma4QuantizedKVCache` instances.
 8. Vulkan: `replace_gemma_sdpa_with_custom_op` AFTER `setup_caches` (it
    classifies layers by whether they own a cache);
    `replace_gemma_rope_with_paired_fused` AFTER that (it re-wraps the
    forward the SDPA transform just installed).
 9. The Vulkan fp16 retarget runs AFTER quantize (there are no torchao
    `rebuild_stack` frames to rewrite before it) and before `setup_caches`.

Plus one that is not about the model: module patches are applied BEFORE
`quantize_`, because `quantize_` only swaps `nn.Linear.weight` for a
quantized subclass and never touches a wrapper's `forward`; patch-then-
quantize is therefore the order-safe direction.

WHAT IS NOT HERE
----------------
No parity harness, no on-device smoke, no phase-A/phase-B split. The
reference drivers interleave those with the build; everything else belongs in a
separate script. The dropped `validate.py`'s pre-flight rules are prose in the
model's NOTES.md.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.export import Dim, ExportedProgram
from torch.export import export as torch_export

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
    XnnpackPartitioner,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

# Side-effect import that registers the `et_vk::gemma_sdpa_with_kv_cache` /
# `et_vk::gemma_custom_sdpa` schemas (with eager reference impls). Pure torch,
# cheap and idempotent, so it stays at module scope.
# NOTE: `executorch.runtime` / `executorch.kernels.quantized` are deliberately
# NOT imported here — each `<backend>/export.py` imports them first, in the
# order that matters (Runtime first, or the quantized kernel registration
# silently no-ops and `embedding_{2,4}bit.out` fails at load_method time).
from ..model import custom_sdpa_ops  # noqa: F401

# `executorch.backends.vulkan.*` is NOT imported at module scope. The Vulkan
# backend is only present in an ExecuTorch built from source with the patched
# `backends/vulkan/` kernels; importing it here would make the XNNPACK export
# unrunnable on a stock ExecuTorch wheel. Every Vulkan symbol is imported
# inside the branch that uses it — `_register_et_vk_ops`, `_build_partitioners`
# and `_setup_caches_and_transforms`.

from ..model.audio import RMSNorm as AudioRMSNorm
from ..model.text import Gemma4Attention, Gemma4EdgeForCausalLM
from ..model.text import RMSNorm as TextRMSNorm
from ..model.transforms.kv_quant import replace_kv_cache_with_quantized_kv_cache
from ..model.transforms.kv_ring import replace_sliding_kv_cache_with_ring_kv_cache
from ..model.transforms.sdpa import replace_gemma_sdpa_with_custom_op
from ..model.truncated import load_truncated_from_safetensors

from .config import Recipe
from .patches import apply_patches, apply_scalar_clamp_patch
from .quant import (
    quantize_module_8da4w,
    quantize_text_model,
    retarget_quantized_to_fp16_with_fp32_scales,
)
from .wrappers import (
    AudioEncoderWavExport,
    SingleMethodForwardExport,
    TextDecoderExport,
    TokenEmbeddingExport,
    VisionEncoderExport,
)

__all__ = ["export"]


# ---------------------------------------------------------------------------
# Constants that are genuinely fixed, not recipe knobs
# ---------------------------------------------------------------------------

# The KV cache / attention-mask dtype passed to `setup_caches`. All four
# drivers hardcode fp16 here regardless of the weight dtype: the XNNPACK
# recipes load at fp16 already, and the Vulkan recipes have been retargeted
# from bf16 to fp16 by the time this runs (constraint 9 above).
_CACHE_DTYPE = torch.float16

# Sequence length of the EXAMPLE inputs handed to `torch.export`. The dim is
# dynamic with min=1, so the traced graph does not depend on this value; it
# only has to lie inside the declared bounds.
#
# UNCERTAIN: `LengthSpec.prefill_len`'s docstring says it is "also the
# example-input length used at trace time", but none of the four drivers do
# that. The XNNPACK ones trace at 128 against a bound of 1024; the Vulkan ones
# trace at 16 against bounds of 128 and 2048. 16 is the cheapest of the two
# observed values and is what both Vulkan drivers use, so it is what we use.
# If a future recipe finds a shape guard that only fires at a longer example,
# this is the knob to raise.
_TRACE_SEQ_LEN = 16

# `k` used for the audio_encoder trace example. 2 is deliberate: the audio
# buffers are baked at K_MAX, so tracing at a SMALL k exercises the runtime
# slice-shrink path most aggressively. (The XNNPACK recipe traces at k=10; the
# graph is the same because the samples dim is dynamic.)
_AUDIO_TRACE_K = 2

# Token id looked up for the vision method's per-layer PLE constants. The
# multimodal contract puts `0` (pad) at every spliced position, never the
# placeholder id — see the model's MODEL_INTERFACE.md "The splice".
_PAD_TOKEN_ID = 0

# Mutable buffers whose initial contents must be baked into the PTE rather
# than left uninitialised.
#
# The text-only XNNPACK recipe lists only the first four despite also enabling
# `use_shared_ring_state`, which is what creates the last two. That looks
# like an oversight rather than a decision, and the pass matches patterns as
# substrings against placeholder names, so naming a buffer that a given
# recipe does not have is a no-op. We always pass all six.
_INIT_MUTABLE_BUFFERS = [
    "kv_cache_pos",
    "k_cache",
    "v_cache",
    "cache_positions",
    "_shared_ring_cache_positions",
    "_shared_ring_kv_cache_pos",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _register_et_vk_ops() -> None:
    """Import the et_vk custom-op library, which registers `apply_rotary_emb`
    and the rest of the `et_vk::` schemas named in the Vulkan `preserve_ops`
    list. Vulkan-only, so it is deferred out of module scope: a stock
    ExecuTorch wheel has no `backends.vulkan` and the XNNPACK export must not
    need one. Cheap and idempotent, so callers need not track it.
    """
    import executorch.backends.vulkan.custom_ops_lib  # noqa: F401


def _human_bytes(n: int) -> str:
    n_f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_f < 1024:
            return f"{n_f:.2f} {unit}"
        n_f /= 1024.0
    return f"{n_f:.2f} PB"


def _say(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


def _resolve_checkpoint(checkpoint: Path) -> Tuple[Path, Path]:
    """Return `(model_dir, safetensors_path)`.

    `model_dir` is what `transformers.from_pretrained` is given (it needs
    config.json + the tokenizer files); `safetensors_path` is what the
    text-decoder loader reads directly.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        st = checkpoint / "model.safetensors"
        if not st.exists():
            candidates = sorted(checkpoint.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(
                    f"no *.safetensors under {checkpoint}"
                )
            st = candidates[0]
        return checkpoint, st
    if checkpoint.suffix == ".safetensors":
        return checkpoint.parent, checkpoint
    raise FileNotFoundError(
        f"checkpoint must be an HF model directory or a .safetensors file, "
        f"got {checkpoint}"
    )


def _uses_hf_vision_tower(recipe: Recipe) -> bool:
    """True when the vision method wraps HuggingFace's `Gemma4VisionEncoder`
    rather than `gemma4.model.vision.Gemma4EdgeVisionTower`.

    The Recipe has no field for this — the two multimodal drivers simply made
    opposite choices, and each choice is tied to its backend: XNNPACK
    monkey-patches HF's vision attention onto `llama.custom_sdpa` and drives
    the encoder layers by hand; Vulkan needs the edge tower because that is
    where the B1/B2/B3/B5 graph-slimming flags and the Vulkan SDPA path live.
    `backend` is therefore the discriminator.
    """
    return recipe.has_vision and recipe.backend == "xnnpack"


# ---------------------------------------------------------------------------
# Stage 1 — class-level flags. MUST run before the model is constructed.
# ---------------------------------------------------------------------------


def _apply_class_flags(recipe: Recipe, *, verbose: bool = True) -> None:
    """Set every CLASS-level attribute the model reads.

    These are class attributes, not instance attributes, so they are process
    global and sticky: a second export in the same process inherits whatever
    the first one set. Every flag is therefore written unconditionally, never
    `if flag: set(True)`.

    `Gemma4Attention.*` is read in `__init__` (cache layout) and in `forward`,
    so this must run before `_load_model`. `RMSNorm.use_rsqrt` is only read in
    `forward`, but there is no reason to treat it differently.
    """
    A = recipe.attn

    Gemma4Attention.use_bshd_kv = A.use_bshd_kv
    Gemma4Attention.use_dynamic_kv_slice = A.use_dynamic_kv_slice
    Gemma4Attention.use_bool_mask = A.use_bool_mask
    Gemma4Attention.use_jit_mask = A.use_jit_mask
    Gemma4Attention.use_shared_ring_state = A.use_shared_ring_state
    Gemma4Attention.skip_redundant_rope_slice = A.skip_redundant_rope_slice
    Gemma4Attention.use_prebroadcast_rope = A.use_prebroadcast_rope

    # `gemma4.model.text.RMSNorm` and `gemma4.model.audio.RMSNorm` are two
    # DIFFERENT classes that happen to share a name. Setting only the text one
    # leaves the audio tower on the `pow(var, -0.5)` path, which XNNPACK
    # refuses (non-integer power), and silently costs the measured −218 host
    # ops / −13.3%. Set both.
    TextRMSNorm.use_rsqrt = A.use_rsqrt
    AudioRMSNorm.use_rsqrt = A.use_rsqrt

    on = [n for n, v in vars(A).items() if v is True]
    _say(verbose, f"[flags] class-level: {', '.join(on) if on else '(none)'}")
    _say(
        verbose,
        f"[flags] RMSNorm.use_rsqrt = {A.use_rsqrt} on BOTH text and audio "
        f"RMSNorm classes",
    )


# ---------------------------------------------------------------------------
# Stage 2 — load the text decoder
# ---------------------------------------------------------------------------


def _rewire_kv_sharing(model: Gemma4EdgeForCausalLM, num_kv_shared: int) -> None:
    """Re-derive the per-attention KV-sharing wiring for a different
    `num_kv_shared_layers` than the loaded config produced.

    Only the flag/`kv_source` half of the original `_truncate_and_wire` is
    kept — the layer-list truncation is already done by the loader.

    Deliberately does NOT rebuild the FFNs. `config.num_kv_shared_layers`
    also selects the double-wide MLP boundary via
    `Gemma4EdgeConfig.ffn_intermediate_size`, and the projections were sized
    from the config the model was CONSTRUCTED with. Changing the wiring after
    the fact keeps the checkpoint-shaped FFN widths, which is what the Vulkan
    recipes rely on when they truncate a full model in place.
    """
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    if not 0 <= num_kv_shared <= n_layers:
        raise ValueError(
            f"num_kv_shared_layers={num_kv_shared} out of range for "
            f"{n_layers} layers"
        )
    cfg.num_kv_shared_layers = num_kv_shared

    first_shared = n_layers - num_kv_shared
    prev_types = cfg.layer_types[:first_shared]
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        ltype = cfg.layer_types[i]
        attn.is_kv_shared_layer = num_kv_shared > 0 and i >= first_shared
        if attn.is_kv_shared_layer:
            attn.kv_shared_layer_index = (
                len(prev_types) - 1 - prev_types[::-1].index(ltype)
            )
            attn.store_full_length_kv = False
        else:
            attn.kv_shared_layer_index = None
            attn.store_full_length_kv = (
                ltype in prev_types
                and i == len(prev_types) - 1 - prev_types[::-1].index(ltype)
            )

    for layer in model.model.layers:
        attn = layer.self_attn
        # `kv_source` is stored with object.__setattr__ so nn.Module does not
        # register it as a submodule — registering it would duplicate every
        # q/k/v/o_proj weight in the state_dict under `...kv_source....`.
        if attn.is_kv_shared_layer:
            object.__setattr__(
                attn,
                "kv_source",
                model.model.layers[attn.kv_shared_layer_index].self_attn,
            )
        else:
            object.__setattr__(attn, "kv_source", None)


def _load_model(
    recipe: Recipe, checkpoint: Path, *, verbose: bool = True
) -> Gemma4EdgeForCausalLM:
    """Load the text decoder at `recipe.quant.base_dtype` and split the PLE.

    `load_truncated_from_safetensors` is used for every N, including the full
    35: at N=35 it is equivalent to `load_from_safetensors` (the layer filter
    and the two PLE slices are all no-ops, and `_truncated_config` rebuilds
    the same config), and below 35 it skips reading the dropped layers off
    disk entirely instead of loading them and throwing them away, which is
    what the Vulkan recipes do.

    `split_ple_for_export()` lives here rather than in `_quantize` because it
    is not a quantisation step: `embed_for_export`, `decode_from_embeds` and
    the inline KV-shared forward all index `_ple_per_layer_embs[i]` and have
    no un-split fallback. It still satisfies constraints 3 and 4 — after
    load_state_dict, before quantize.

    UNCERTAIN (truncated smoke tests only, N < 35): the two loaders disagree
    about the last layer of a truncation. `load_truncated_from_safetensors`
    builds a fresh `Gemma4EdgeConfig`, whose `__post_init__` forces
    `layer_types[-1] = "full_attention"`; the in-place
    `_truncate_and_wire` slices `cfg.layer_types` and leaves the last entry
    as-is. They agree at N=35 (layer 34 is already full_attention) and can
    differ for a truncated Vulkan smoke.
    """
    L = recipe.lengths
    _, safetensors = _resolve_checkpoint(checkpoint)

    t0 = time.perf_counter()
    _say(
        verbose,
        f"[load] {safetensors} at {recipe.quant.base_dtype}, "
        f"N={L.num_layers} layers",
    )
    model = load_truncated_from_safetensors(
        str(safetensors),
        num_layers=L.num_layers,
        dtype=recipe.quant.base_dtype,
    ).eval()

    derived = model.config.num_kv_shared_layers
    if derived != L.num_kv_shared_layers:
        _say(
            verbose,
            f"[load] rewiring KV sharing {derived} -> "
            f"{L.num_kv_shared_layers} shared layers",
        )
        _rewire_kv_sharing(model, L.num_kv_shared_layers)

    # Constraint 3. Also drops `embed_tokens_per_layer` /
    # `per_layer_model_projection` so they cannot be duplicated in the graph.
    if model.config.use_ple:
        model.model.split_ple_for_export()
        _say(
            verbose,
            f"[load] split_ple_for_export: "
            f"{len(model.model._ple_per_layer_embs)} per-layer PLE modules",
        )

    _say(verbose, f"[load] done in {time.perf_counter() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Stage 3 — instance-level attention flags
# ---------------------------------------------------------------------------


def _apply_instance_flags(
    model: Gemma4EdgeForCausalLM, recipe: Recipe, *, verbose: bool = True
) -> None:
    """Set the two PER-ATTENTION flags, which are instance attributes and so
    must be set after construction.

    `use_broadcast_gqa` is gated on `kv_group_size > 1` exactly as the XNNPACK
    recipes gate it: with one KV head per Q head there is nothing to broadcast
    and the flag would only change which code path emits the same maths.
    """
    A = recipe.attn
    if A.use_custom_sdpa:
        # Registers `torch.ops.llama.custom_sdpa`. Deferred to here (it loads
        # a shared library) and only done when a recipe actually asks for the
        # fused op — the Vulkan recipes use the et_vk ops instead.
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401

    n_attn = n_gqa = 0
    for m in model.modules():
        if not isinstance(m, Gemma4Attention):
            continue
        n_attn += 1
        m.use_custom_sdpa = A.use_custom_sdpa
        if A.use_broadcast_gqa and m.kv_group_size > 1:
            m.use_broadcast_gqa = True
            n_gqa += 1
        else:
            m.use_broadcast_gqa = False

    _say(
        verbose,
        f"[flags] instance-level on {n_attn} attentions: "
        f"use_custom_sdpa={A.use_custom_sdpa}, "
        f"use_broadcast_gqa on {n_gqa}",
    )


# ---------------------------------------------------------------------------
# Stage 4 — quantisation (+ the Vulkan fp16 retarget)
# ---------------------------------------------------------------------------


def _quantize(
    model: Gemma4EdgeForCausalLM, recipe: Recipe, *, verbose: bool = True
) -> Gemma4EdgeForCausalLM:
    """8da4w linears + weight-only embeddings, then (Vulkan) the fp16 retarget.

    Returns the model: `quantize_text_model` ends in `unwrap_tensor_subclass`,
    which may hand back a different object.
    """
    Q = recipe.quant
    t0 = time.perf_counter()
    _say(
        verbose,
        f"[quant] linears int{Q.linear_bits} g{Q.linear_group_size}"
        + (
            f" skip={list(Q.skip_linear_name_suffixes)}"
            if Q.skip_linear_name_suffixes
            else ""
        )
        + f"; main emb g{Q.embedding_group_size}"
        f"; PLE int{Q.ple_embedding_bits} g{Q.ple_embedding_group_size}",
    )

    kwargs = dict(
        linear_bits=Q.linear_bits,
        linear_group_size=Q.linear_group_size,
        dynamic_activation=Q.linear_dynamic_activation,
        skip_linear_name_suffixes=Q.skip_linear_name_suffixes,
        quantize_main_embeddings=Q.quantize_main_embeddings,
        embedding_group_size=Q.embedding_group_size,
        quantize_ple_embeddings=Q.quantize_ple_embeddings,
        ple_embedding_bits=Q.ple_embedding_bits,
        ple_embedding_group_size=Q.ple_embedding_group_size,
    )
    try:
        model = quantize_text_model(model, **kwargs)
    except Exception as exc:  # noqa: BLE001
        # torchao occasionally refuses a bf16 input tensor; the recorded
        # workaround is a fp32 round-trip.
        # Only the bf16 (Vulkan) lineage has ever hit this, and letting an
        # fp16 failure through unmodified is the correct behaviour for the
        # XNNPACK lineage, which has no such fallback.
        if Q.base_dtype != torch.bfloat16:
            raise
        _say(verbose, f"[quant] bf16 quantize failed ({exc!r}); fp32 round-trip")
        model = model.to(torch.float32)
        model = quantize_text_model(model, **kwargs)
        try:
            model = model.to(torch.bfloat16)
        except Exception as exc2:  # noqa: BLE001
            _say(verbose, f"[quant] cast back to bf16 failed ({exc2!r}); staying fp32")

    _say(verbose, f"[quant] quantize_model in {time.perf_counter() - t0:.1f}s")

    if Q.fp16_retarget:
        # Constraint 9. Rewrites each torchao `rebuild_stack` frame's dtype to
        # fp16 (so the reconstructed wrapper dispatches its matmul on fp16,
        # which is what the Vulkan `linear_q4gsw_*` shaders want) while
        # upcasting the per-group scales bf16 -> fp32 (lossless; the bf16
        # mantissa fits exactly). bf16 -> fp16 rounding of those scales was
        # the dominant fp16 accuracy floor; with fp32 scales the on-device
        # decoder reaches cos 0.99 vs HF bf16.
        t0 = time.perf_counter()
        retarget_quantized_to_fp16_with_fp32_scales(model)
        _say(
            verbose,
            f"[quant] fp16 retarget with fp32 scales in "
            f"{time.perf_counter() - t0:.1f}s",
        )

    return model


# ---------------------------------------------------------------------------
# Stage 5 — caches and source transforms
# ---------------------------------------------------------------------------


def _setup_caches_and_transforms(
    model: Gemma4EdgeForCausalLM, recipe: Recipe, *, verbose: bool = True
) -> Gemma4EdgeForCausalLM:
    """Allocate the KV caches, then apply the cache swaps / source transforms.

    Constraints 5-8 all live in this function, in order.
    """
    A, L, Q = recipe.attn, recipe.lengths, recipe.quant

    _say(
        verbose,
        f"[caches] setup_caches(max_seq_len={L.max_seq_len}, "
        f"dtype={_CACHE_DTYPE})",
    )
    model.setup_caches(batch_size=1, max_seq_len=L.max_seq_len, dtype=_CACHE_DTYPE)

    if A.int8_kv_cache:
        # Constraint 6. Bit-exact vs fp32 in eager across four prompts; cuts a
        # 16k-position KV buffer from ~900 MB to ~240 MB.
        replace_kv_cache_with_quantized_kv_cache(model)
        _say(verbose, "[caches] replace_kv_cache_with_quantized_kv_cache")

    if A.ring_kv_cache:
        # Constraint 7. Must follow the int8 swap so it wraps the
        # `Gemma4QuantizedKVCache` instances, not the plain float ones. Also
        # clears the dead `[max_seq_len, max_seq_len]` mask buffer on the
        # sliding layers.
        replace_sliding_kv_cache_with_ring_kv_cache(model)
        _say(verbose, "[caches] replace_sliding_kv_cache_with_ring_kv_cache")

    if A.vulkan_sdpa_transform:
        # Constraint 8, first half. Raises rather than degrading if it meets
        # ring KV / use_dynamic_kv_slice / use_bshd_kv=False / caches that
        # were never set up — NOTES.md records these as rules R1-R3.
        replace_gemma_sdpa_with_custom_op(model)
        _say(
            verbose,
            f"[transform] gemma_sdpa: "
            f"{getattr(model, '_gemma_sdpa_transform_stats', {})}",
        )

    if A.vulkan_rope_transform:
        # Constraint 8, second half: this re-wraps the forward the SDPA
        # transform just installed, so it cannot run first. Consumer
        # (KV-shared) layers are skipped — the fused op needs both q and k,
        # and consumers have no k projection to feed it.
        #
        # Imported here rather than at module scope: `transforms/rope.py`
        # pulls in `executorch.backends.vulkan.custom_ops_lib`.
        from ..model.transforms.rope import replace_gemma_rope_with_paired_fused

        replace_gemma_rope_with_paired_fused(model)
        _say(
            verbose,
            f"[transform] paired RoPE: {getattr(model, '_paired_rope_stats', {})}",
        )

    # Export-time output knobs. Plain attributes read inside forward, so their
    # position relative to the transforms does not matter; set here so every
    # model-shaping decision is in one place.
    #
    # return_last_only=True slices hidden to the final position BEFORE the
    # lm_head, dropping logits from [B, S, vocab] to [B, 1, vocab]. The text
    # recipes leave it False so the runner can read the last-position slice
    # itself after a prefill chunk (text_decoder_runner.cpp does
    # `logits += (num_tokens - 1) * vocab_size`).
    model.return_last_only = A.return_last_only
    model.logits_dtype = Q.logits_dtype
    _say(
        verbose,
        f"[model] return_last_only={A.return_last_only} "
        f"logits_dtype={Q.logits_dtype}",
    )

    model.reset_caches()
    return model


# ---------------------------------------------------------------------------
# Stage 6 — vision / audio towers
# ---------------------------------------------------------------------------


def _build_edge_vision_tower(
    hf_vision_tower: nn.Module, recipe: Recipe, *, verbose: bool = True
) -> nn.Module:
    """Rebuild HF's vision tower as `Gemma4EdgeVisionTower` with the Vulkan
    graph-slimming flags on.

    B1 `use_embedding_position_emb`  one-hot @ table -> two F.embedding
        gathers. Removes a 23 MB ephemeral and a wide matmul shader.
    B2 `use_reshape_pool`            pooler one-hot matmul -> avg_pool2d.
        Needs the static grid dims, which is why they are set from the recipe.
    B3 `use_static_mask`             bake the all-zeros bidirectional mask.
        Also a precondition for B5.
    B4 `use_fused_sdpa`              OFF. `F.scaled_dot_product_attention`
        decomposes on Vulkan into 32x `where.self` on fp32 (1,12,1120,1120)
        plus 32x `expand_copy` per layer and doubles the arena 91 -> 184 MB.
    B5 `use_vulkan_sdpa`             one `et_vk::gemma_custom_sdpa` per layer.
        Took the vision method from 36 delegates / 66 MB to 4 / 10 MB.

    `pooling_kernel_size` is deliberately left at whatever the HF config says:
    `_avg_pool_by_positions` derives k from `input_seq_len // output_length`,
    so `LengthSpec.pool_k` reaches it through `max_soft_tokens`.
    """
    from ..model.vision import (
        Gemma4EdgeVisionTower,
        _config_from_hf,
        _load_state_from_hf,
    )

    L = recipe.lengths
    cfg = _config_from_hf(hf_vision_tower.config)
    cfg.use_embedding_position_emb = True   # B1
    cfg.use_reshape_pool = True             # B2
    cfg.patch_grid_nx = L.patch_grid_nx
    cfg.patch_grid_ny = L.patch_grid_ny
    cfg.use_static_mask = True              # B3
    cfg.use_fused_sdpa = False              # B4 — see docstring
    cfg.use_vulkan_sdpa = True              # B5

    tower = Gemma4EdgeVisionTower(cfg)
    _load_state_from_hf(tower, hf_vision_tower)
    tower.eval()
    # B3 bakes the (1, 1, P, P) zeros mask at the known patch count; B5 needs
    # per-layer (k_cache_scratch, v_cache_scratch) materialised at that same
    # count. Both are no-ops when their flag is off.
    tower.bake_static_mask(num_patches=L.max_patches)
    tower.bake_sdpa_caches(num_patches=L.max_patches)
    _say(
        verbose,
        f"[vision] edge tower: {sum(p.numel() for p in tower.parameters())/1e6:.1f}M "
        f"params, B1+B2+B3+B5, {L.max_patches} patches",
    )
    return tower


def _build_towers(
    recipe: Recipe, checkpoint: Path, *, verbose: bool = True
) -> Tuple[Optional[Tuple[nn.Module, nn.Module]], Optional[nn.Module]]:
    """Return `(vision, audio)`.

    `vision` is `(vision_tower, embed_vision)` — the wrapper that turns them
    into a PTE method is built in `_build_programs`, because it needs the
    text model too (for the baked per-layer PLE constants).
    `audio` is a ready-to-export `AudioEncoderWavExport`.
    Both are None for a text-only recipe.

    One HF load serves both towers. The XNNPACK recipe does this; the Vulkan
    one pays the load twice (once in its vision factory, once in its audio
    factory) purely because the two factories were written independently.

    NOTE (audio): the shipping recipes wrap the STOCK HF `Gemma4AudioTower`,
    not `gemma4.model.audio.Gemma4EdgeAudioTower`. An earlier attempt with the
    edge tower produced uniform-noise logits on every modality and was reverted
    to the HF path, which is what ships. Consequently the
    edge tower's `bake_*` flags — which must be written to FOUR separate
    config objects (`edge.config`, `edge.audio_tower.config`,
    `layer.config`, `layer.self_attn.config`, because each module captured its
    own reference) — are not exercised here at all. If an edge audio tower is
    ever reinstated, that four-way write is the trap, and
    `wrappers.AudioEncoderEdgeTowerExport` is the wrapper to use.
    """
    if not (recipe.has_vision or recipe.has_audio):
        return None, None

    Q = recipe.quant
    model_dir, _ = _resolve_checkpoint(checkpoint)

    from transformers import Gemma4Config
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4AudioModel, Gemma4MultimodalEmbedder, Gemma4VisionModel,
    )
    from safetensors import safe_open

    t0 = time.perf_counter()
    _say(verbose, f"[towers] loading HF vision/audio towers (fp32) from {model_dir}")
    # Construct only the encoders we use. Loading a second, fp32 text model
    # before discarding it requires another ~20 GB during an omni export.
    config = Gemma4Config.from_pretrained(str(model_dir), local_files_only=True)
    hf = nn.Module()
    hf.model = nn.Module()
    hf.model.language_model = None
    if recipe.has_vision:
        hf.model.vision_tower = Gemma4VisionModel(config.vision_config)
        hf.model.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
    if recipe.has_audio:
        hf.model.audio_tower = Gemma4AudioModel(config.audio_config)
        hf.model.embed_audio = Gemma4MultimodalEmbedder(config.audio_config, config.text_config)
    expected = set(hf.state_dict())
    state = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as tensors:
            for key in tensors.keys():
                if key in expected:
                    value = tensors.get_tensor(key)
                    state[key] = value.float() if value.is_floating_point() else value
    hf.load_state_dict(state, strict=True, assign=True)
    hf.eval()
    del state
    gc.collect()
    _say(verbose, f"[towers] HF load: {time.perf_counter() - t0:.1f}s")

    vision: Optional[Tuple[nn.Module, nn.Module]] = None
    if recipe.has_vision:
        hf_vision_tower = hf.model.vision_tower.eval()
        embed_vision = hf.model.embed_vision.eval()
        if _uses_hf_vision_tower(recipe):
            # XNNPACK: the HF tower is used as-is. Its per-layer attention
            # forward has already been replaced process-wide by patch P6
            # (`llama.custom_sdpa`) and its patch embedder by P5, both applied
            # at the top of `export()`.
            vision_tower = hf_vision_tower
            # P7 — scalar clamp. The HF tower's Gemma4ClippableLinear modules
            # hold their QAT bounds as 0-d Tensor buffers, which trace to
            # `aten.clamp.Tensor`. Rebind to Python-float bounds
            # (`aten.clamp.default`). Per-instance, so it cannot live in
            # `apply_patches()`; must run BEFORE quantize_ (patch-then-quantize
            # is the order-safe direction) and before tracing. Eager bit-exact.
            n_clamp = apply_scalar_clamp_patch(vision_tower)
            n_clamp += apply_scalar_clamp_patch(embed_vision)
            _say(
                verbose,
                f"[vision] HuggingFace Gemma4VisionEncoder "
                f"(patched P5+P6, P7 on {n_clamp} ClippableLinear)",
            )
        else:
            vision_tower = _build_edge_vision_tower(
                hf_vision_tower, recipe, verbose=verbose
            )
            del hf_vision_tower

        if Q.vision_quant == "8da4w":
            # The XNNPACK recipe quantises its assembled wrapper; the Vulkan
            # one quantises the tower and embed_vision
            # separately. Same result: the wrapper only holds references to
            # sub-modules of these two plus non-Linear std_* buffers.
            vision_tower = quantize_module_8da4w(
                vision_tower, group_size=Q.vision_group_size
            )
            embed_vision = quantize_module_8da4w(
                embed_vision, group_size=Q.vision_group_size
            )
            _say(verbose, f"[vision] 8da4w PerGroup({Q.vision_group_size})")
        else:
            # No fp16 retarget for vision even on Vulkan. Retargeting measured
            # a 5x magnitude inflation in the vision output (cos 0.989 but
            # L2/pos 142 vs eager 27.7), bisected to the fp32 -> fp16 cast of
            # bf16-trained vision params. Vision stays fp32 end to end; it is
            # ~3.5 GB of the 4.37 GB PTE and the largest open size lever.
            _say(verbose, "[vision] fp32, unquantised, no fp16 retarget")

        vision = (vision_tower, embed_vision)

    audio: Optional[nn.Module] = None
    if recipe.has_audio:
        audio_tower = hf.model.audio_tower.eval()
        embed_audio = hf.model.embed_audio.eval()
        if Q.audio_quant == "8da4w":
            # 8da4w on the tower's nn.Linear only (conv2d weights and
            # the frontend hann/dft/mel buffers stay fp32). 4.07 -> 3.08 GB.
            # The audio eager quant floor vs fp32 is cos ~0.92 — that is the
            # AUDIO number; 0.95-0.97 is the text one.
            audio_tower = quantize_module_8da4w(
                audio_tower, group_size=Q.audio_group_size
            )
            _say(verbose, f"[audio] 8da4w PerGroup({Q.audio_group_size})")
        else:
            # Audio weights stay fp32. An fp16 audio PTE measured min_cos 0.85
            # on a row and flipped the first-token argmax 818 -> 100, despite
            # eager fp16 measuring 0.99996.
            _say(verbose, "[audio] stock HF Gemma4AudioTower, fp32 weights")

        # Output cast: Vulkan appends `.to(fp16)` so the host splice into the
        # text_decoder's fp16 embeds buffer is a clean memcpy; XNNPACK returns
        # the encoder dtype unchanged. This is the
        # documented runtime contract in MODEL_INTERFACE.md — "on the
        # Vulkan path the audio encoder output is Half; on fp32 paths float".
        out_dtype = torch.float16 if recipe.backend == "vulkan" else None
        audio = AudioEncoderWavExport(
            audio_tower, embed_audio, output_dtype=out_dtype
        ).eval()
        _say(verbose, f"[audio] output_dtype={out_dtype}")

    hf.model.vision_tower = None
    hf.model.embed_vision = None
    hf.model.audio_tower = None
    hf.model.embed_audio = None
    del hf
    gc.collect()

    return vision, audio


# ---------------------------------------------------------------------------
# Stage 7 — torch.export, one program per PTE method
# ---------------------------------------------------------------------------


def _build_programs(
    model: Gemma4EdgeForCausalLM,
    towers: Tuple[Optional[Tuple[nn.Module, nn.Module]], Optional[nn.Module]],
    recipe: Recipe,
    *,
    verbose: bool = True,
) -> dict:
    """Trace every PTE method. Returns `{method_name: ExportedProgram}`.

    `dynamic_shapes` is keyed by PARAMETER NAME, so these dicts are locked to
    the wrapper signatures in `wrappers.py`. Those standardised on the Vulkan
    recipe's names (`embeds`, `ple_tokens`, `input_pos`) — the older
    `inputs_embeds` / `ple_tok` keys would silently fail to bind.
    """
    L = recipe.lengths
    vision, audio = towers
    programs: dict[str, ExportedProgram] = {}

    example_len = max(1, min(_TRACE_SEQ_LEN, L.prefill_len))
    example_ids = torch.zeros(1, example_len, dtype=torch.long)
    example_pos = torch.arange(example_len, dtype=torch.long)

    if recipe.modality == "text":
        # One method, and it MUST be called "forward":
        # extension/llm/runner/text_decoder_runner.h hardcodes
        # load_method("forward"), and a PTE whose only method has another name
        # SIGABRTs at load.
        #
        # `use_inline_kv_shared_loop` re-implements the causal-LM forward with
        # explicit KV-sharing threading. Its original reason — that
        # `tensor[input_pos]` lowers to `aten.index.Tensor`, which the Vulkan
        # partitioner rejects and which splits the decoder out of its single
        # partition — has since been folded into `Gemma4EdgeModel.forward`
        # itself, so the False path is also Vulkan-clean for RoPE now. What
        # still differs on the True path is that it skips the
        # `use_prebroadcast_rope` block and never passes ring state to layers.
        # Kept on for Vulkan because that is the graph validated on device.
        wrapped = SingleMethodForwardExport(
            model, use_inline_kv_shared_loop=(recipe.backend == "vulkan")
        ).eval()

        # Cheap recipe smoke before paying for an export.
        # A NaN here means the flag combination is broken, and finding that
        # out now instead of forty minutes later is worth one forward.
        model.reset_caches()
        with torch.no_grad():
            probe = wrapped(example_ids, example_pos)
        n_nan = int(torch.isnan(probe).sum())
        n_inf = int(torch.isinf(probe).sum())
        _say(
            verbose,
            f"[export] eager probe S={example_len}: logits "
            f"{tuple(probe.shape)} {probe.dtype} NaN={n_nan} Inf={n_inf}",
        )
        if n_nan or n_inf:
            raise RuntimeError(
                "eager probe produced NaN/Inf before export — the recipe is "
                "broken; do not trust a PTE built from it"
            )
        del probe
        model.reset_caches()

        # min=1 so the single method serves decode (S=1) as well as prefill.
        seq_dim = Dim("seq_len", min=1, max=L.prefill_len)
        t0 = time.perf_counter()
        programs["forward"] = torch_export(
            wrapped,
            (example_ids, example_pos),
            strict=True,
            dynamic_shapes={"input_ids": {1: seq_dim}, "input_pos": {0: seq_dim}},
        )
        _say(verbose, f"[export] forward: {time.perf_counter() - t0:.1f}s")
        model.reset_caches()
        return programs

    # ---- multimodal: four methods -----------------------------------------
    #
    # token_embedding and text_decoder get INDEPENDENT dynamic dims. That
    # independence is the whole reason the multimodal PTE splits into two
    # methods: token_embedding must cover the full multimodal id sequence
    # (prefill_len), while text_decoder's dim is what the activation arena
    # scales with and is held as low as the encoders allow.
    seq_dim = Dim("seq_len", min=1, max=L.prefill_len)
    text_dec_seq_dim = Dim(
        "text_dec_seq_len", min=1, max=L.effective_text_decoder_prefill_len
    )

    token_embedder = TokenEmbeddingExport(model).eval()
    text_decoder = TextDecoderExport(model).eval()

    model.reset_caches()
    with torch.no_grad():
        example_embeds, example_ple = model.embed_for_export(example_ids)

    t0 = time.perf_counter()
    programs["token_embedding"] = torch_export(
        token_embedder,
        (example_ids,),
        strict=True,
        dynamic_shapes={"input_ids": {1: seq_dim}},
    )
    _say(verbose, f"[export] token_embedding: {time.perf_counter() - t0:.1f}s")

    # text_decoder owns the KV cache, so reset on both sides of the trace:
    # a cache left dirty by the trace leaks into whatever runs next, and one
    # left dirty by an earlier step would be baked in as the initial state.
    model.reset_caches()
    t0 = time.perf_counter()
    programs["text_decoder"] = torch_export(
        text_decoder,
        (example_embeds, example_ple, example_pos),
        strict=True,
        dynamic_shapes={
            "embeds": {1: text_dec_seq_dim},
            "ple_tokens": {1: text_dec_seq_dim},
            "input_pos": {0: text_dec_seq_dim},
        },
    )
    _say(
        verbose,
        f"[export] text_decoder: {time.perf_counter() - t0:.1f}s "
        f"(seq bound {L.effective_text_decoder_prefill_len})",
    )
    model.reset_caches()

    if vision is not None:
        vision_tower, embed_vision = vision
        hf_style = _uses_hf_vision_tower(recipe)
        vision_mod = VisionEncoderExport(
            vision_tower,
            embed_vision,
            vis_nx=L.patch_grid_nx,
            vis_ny=L.patch_grid_ny,
            patch_size=L.patch_size,
            pool_k=L.pool_k,
            max_soft_tokens=L.max_soft_tokens,
            # XNNPACK drives `encoder.layers` by hand with attention_mask=None
            # because P6 replaced each layer's attention forward wholesale and
            # it ignores the mask; Vulkan calls the encoder module so the
            # edge tower's static-mask bake and Vulkan SDPA path are used.
            encoder_style="layer_loop" if hf_style else "module",
            # Vulkan only: bake `ple_layer(pad_ids)` per layer as a constant
            # (the input is constant, so the output is too). Drops 35x
            # aten.embedding + 35x dequantize_affine from this method and
            # keeps the int2 PLE tables in text_decoder only. The XNNPACK
            # vision method returns embeds alone.
            emit_ple_tokens=not hf_style,
            lm=None if hf_style else model,
            pad_token_id=_PAD_TOKEN_ID,
        ).eval()

        # Fully static: no dynamic_shapes. The runner must hand this method a
        # CONTIGUOUS fp32 (1, 3, H, W) tensor in the raw 0-255 range — the
        # 1/255 rescale and the patchify unfold both live inside the graph,
        # and ExecuTorch reads a non-contiguous view's raw buffer against the
        # PTE's expected strides, yielding cos ~0.7 with no error.
        image_example = torch.zeros(1, 3, L.vision_h, L.vision_w, dtype=torch.float32)
        t0 = time.perf_counter()
        programs["vision_encoder"] = torch_export(
            vision_mod, (image_example,), strict=True
        )
        _say(
            verbose,
            f"[export] vision_encoder: {time.perf_counter() - t0:.1f}s "
            f"({L.vision_h}x{L.vision_w} -> {L.max_soft_tokens} soft tokens)",
        )

    if audio is not None:
        # The waveform dim is declared as a MULTIPLE of the block size —
        # `BLOCK_SAMPLES * Dim(...)` — not as a free range, because
        # MelFrontend emits exactly 48*k - 1 frames for 7680*k samples and
        # the wrapper pads one frame to reach the chunked-attention block
        # boundary. Any other length breaks that identity.
        k_dim = Dim("audio_k", min=L.audio_k_min, max=L.audio_k_max)
        samples_dim = L.audio_block_samples * k_dim
        trace_k = min(max(_AUDIO_TRACE_K, L.audio_k_min), L.audio_k_max)
        n_trace = L.audio_block_samples * trace_k
        waveform_example = torch.zeros(1, n_trace, dtype=torch.float32)
        num_valid_example = torch.tensor(n_trace, dtype=torch.long)
        t0 = time.perf_counter()
        programs["audio_encoder"] = torch_export(
            audio,
            (waveform_example, num_valid_example),
            strict=True,
            dynamic_shapes={
                "waveform": {1: samples_dim},
                # 0-d scalar; no dynamic dim to declare.
                "num_valid_samples": None,
            },
        )
        _say(
            verbose,
            f"[export] audio_encoder: {time.perf_counter() - t0:.1f}s "
            f"(k in [{L.audio_k_min}, {L.audio_k_max}], traced at k={trace_k})",
        )

    # Emit in the order `Recipe.method_names` declares, so the PTE's execution
    # plan order is a property of the recipe rather than of this function.
    # (The two multimodal recipes disagree on the middle two; nothing depends
    # on it.)
    return {m: programs[m] for m in recipe.method_names if m in programs}


# ---------------------------------------------------------------------------
# Stage 8 — partitioners
# ---------------------------------------------------------------------------


def _build_partitioners(recipe: Recipe) -> dict:
    """Return `{method_name: [partitioner, ...]}`.

    XNNPACK: `XnnpackDynamicallyQuantizedPartitioner` FIRST, then
    `XnnpackPartitioner`. The order is load-bearing — with the plain
    partitioner first, it const-folds the dequantised weights into its
    delegate blob and the 8da4w size saving vanishes entirely (measured
    1.00x, i.e. none). Fresh instances per method, as the XNNPACK recipes
    build them.

    Vulkan: ONE `VulkanPartitioner` instance shared across every Vulkan
    method, exactly as the Vulkan recipes do. No storage-type override, no
    memory-layout override, no fp16 flag, no allowlist.
      - `require_dynamic_shapes=True` makes the partitioner refuse ops whose
        Vulkan implementation has no `resize_fn`, instead of silently
        producing a delegate that computes against stale shapes.
      - `torchao.dequantize_affine` is blocklisted so it stays on the host,
        where `do_quant_fusion_and_const_prop` can rewrite the
        dequantize+embedding pattern into the packed
        `quantized_decomposed.embedding_{byte,4bit}` kernels.

    `audio_encoder` runs on XNNPACK even inside an otherwise-Vulkan PTE: the
    Vulkan `gemma_custom_sdpa` per-head + softcap shader measured cos 0.40
    against eager, and every recipe since reroutes it.
    NOTES.md records this as rule R10: `audio_backend="vulkan"` is refused.
    """

    def _xnnpack_pair() -> list:
        return [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]

    if recipe.backend == "xnnpack":
        return {m: _xnnpack_pair() for m in recipe.method_names}

    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )

    shared = VulkanPartitioner(
        compile_options={"require_dynamic_shapes": True},
        operator_blocklist={exir_ops.edge.torchao.dequantize_affine.default},
    )
    out: dict = {}
    for method in recipe.method_names:
        if method == "audio_encoder" and recipe.audio_backend == "xnnpack":
            out[method] = _xnnpack_pair()
        else:
            out[method] = [shared]
    return out


# ---------------------------------------------------------------------------
# Stage 9 — lower, serialize
# ---------------------------------------------------------------------------


def _print_partition_diagnostic(edge, method_name: str) -> None:
    """Host-op / delegate census for one lowered method.

    This is the number to read when a Vulkan export "succeeds" but runs slowly
    or OOMs: a method that should be one delegate and is thirty has fallen
    apart at some rejected op, which the top host-op kinds will name.
    """
    try:
        graph = edge.exported_program(method_name).graph_module.graph
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: partition diagnostic [{method_name}] failed ({exc})")
        return

    n_total = n_call = n_delegate = 0
    host_kinds: dict[str, int] = {}
    for node in graph.nodes:
        n_total += 1
        if node.op != "call_function":
            continue
        n_call += 1
        target = str(node.target)
        if "executorch_call_delegate" in target:
            n_delegate += 1
        else:
            host_kinds[target] = host_kinds.get(target, 0) + 1

    host = n_call - n_delegate
    print(f"\n  partition [{method_name}]:")
    print(f"    graph nodes    = {n_total}")
    print(f"    call_function  = {n_call}")
    print(f"    delegate calls = {n_delegate}")
    print(f"    host ops       = {host}")
    if n_call:
        print(f"    host fraction  = {host / n_call * 100.0:.1f}%")
    for kind, count in sorted(host_kinds.items(), key=lambda kv: -kv[1])[:10]:
        print(f"      {count:4d}  {kind}")


def _print_arena_bytes(et_program, method_name: str) -> None:
    """Activation arena for one method.

    Arena bytes — not partition count — is what predicts whether
    `load_method` OOMs on device: a build with 37 partitions / 91 MB OOM'd
    while one with 36 partitions / 66 MB loaded fine.
    """
    try:
        plans = et_program.executorch_program.execution_plan
        plan = next(p for p in plans if p.name == method_name)
        arena = sum(int(b) for b in plan.non_const_buffer_sizes)
        print(f"  arena[{method_name}] = {_human_bytes(arena)} ({arena} bytes)")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: arena diagnostic [{method_name}] failed ({exc})")


def _lower_and_write(
    programs: dict,
    partitioners: dict,
    recipe: Recipe,
    out_path: Path,
    *,
    schema_json: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    """`to_edge_transform_and_lower` -> `to_executorch` -> serialize."""
    if recipe.backend == "vulkan":
        # Registers `et_vk::apply_rotary_emb`, named in `preserve_ops` below.
        _register_et_vk_ops()
        compile_config = EdgeCompileConfig(
            _check_ir_validity=False,
            # Without this the edge-lowering decomposition table shreds the
            # three fused ops back into their reference implementations, and
            # the Vulkan partitioner then sees the decomposed chain instead of
            # the single node it has a shader for.
            preserve_ops=[
                torch.ops.et_vk.gemma_sdpa_with_kv_cache.default,
                torch.ops.et_vk.gemma_custom_sdpa.default,
                torch.ops.et_vk.apply_rotary_emb.default,
            ],
        )
        backend_config = ExecutorchBackendConfig(
            passes=[InitializedMutableBufferPass(_INIT_MUTABLE_BUFFERS)],
            do_quant_fusion_and_const_prop=True,
        )
    else:
        compile_config = EdgeCompileConfig(_check_ir_validity=False)
        backend_config = ExecutorchBackendConfig(
            # Mutable buffers that need their initial contents baked in rather
            # than left uninitialised — the KV cache position counters and the
            # model-level shared ring scratch.
            passes=[InitializedMutableBufferPass(_INIT_MUTABLE_BUFFERS)],
            # Rewrites the leftover `torchao.dequantize_affine + aten.embedding`
            # pattern into `quantized_decomposed.embedding_{byte,4bit}` and
            # materialises the packed int4 weights. Without it the quantised
            # embedding tables are stored dequantised and the size win is lost.
            do_quant_fusion_and_const_prop=True,
        )

    constant_methods = recipe.constant_methods()
    if schema_json is not None:
        # The react-native-executorch runtime reads the model's I/O spec from
        # this constant method: `ModelHostObject` executes `get_model_schema`
        # at load and parses the JSON into a ModelSpec with real dynamic-dim
        # domains and runtime constraints. Without it the runtime falls back to
        # `methodSpecFromMetadata`, where every dim is a static constant taken
        # from the compiled upper bound -- so a decode call at S=1 against a
        # graph traced to S<=1024 is rejected. See src/exporter/schema.py.
        constant_methods["get_model_schema"] = schema_json
    _say(verbose, "\n[lower] constant_methods:")
    for key, value in constant_methods.items():
        _say(verbose, f"    {key:<26} {value!r}")

    t0 = time.perf_counter()
    edge = to_edge_transform_and_lower(
        programs,
        compile_config=compile_config,
        partitioner=partitioners,
        constant_methods=constant_methods,
    )
    _say(
        verbose,
        f"[lower] to_edge_transform_and_lower: {time.perf_counter() - t0:.1f}s",
    )

    if verbose:
        for method_name in programs:
            _print_partition_diagnostic(edge, method_name)

    t0 = time.perf_counter()
    et_program = edge.to_executorch(backend_config)
    _say(verbose, f"\n[lower] to_executorch: {time.perf_counter() - t0:.1f}s")

    if verbose:
        for method_name in programs:
            _print_arena_bytes(et_program, method_name)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(out_path, "wb") as f:
        # write_to_file, NEVER `.buffer`. The buffer property materialises the
        # whole PTE as a Python bytes object and truncates above 2 GiB — every
        # multimodal recipe here is larger than that.
        et_program.write_to_file(f)
    size = out_path.stat().st_size
    _say(
        verbose,
        f"[write] {out_path} — {_human_bytes(size)} in "
        f"{time.perf_counter() - t0:.1f}s",
    )
    if size > 5 * 1024 ** 3:
        _say(
            verbose,
            "[write] WARNING: PTE > 5 GB, which is over the S24 unified-memory "
            "budget. Check for duplicated constants across methods.",
        )
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export(
    recipe: Recipe,
    *,
    checkpoint: Path,
    out_path: Path,
    schema_json: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    """Build, transform, quantize, export, lower and write the PTE.

    Returns out_path.

    `checkpoint` is the HuggingFace model directory (it needs config.json for
    the text config and, for multimodal recipes, the full HF model for the
    vision and audio towers); a direct path to a .safetensors file is also
    accepted for text-only recipes.

    This does not validate the recipe. The dropped `validate.py`'s rules are
    prose in the model's NOTES.md and are NOT enforced in code — check them by
    hand when changing a recipe.
    """
    t_start = time.perf_counter()
    checkpoint = Path(checkpoint)
    out_path = Path(out_path)

    _say(verbose, "=" * 72)
    _say(verbose, f"gemma4.export.pipeline — recipe {recipe.name!r}")
    _say(verbose, "=" * 72)
    _say(verbose, recipe.summary())
    _say(verbose, "")
    _say(verbose, f"checkpoint  {checkpoint}")
    _say(verbose, f"out         {out_path}")
    _say(verbose, "")

    # [1/8] Monkey-patches, before anything is constructed, quantised or
    # traced. P1/P3 mutate the decomposition table and one edge-lowering
    # passes, so they must land before `to_edge_transform_and_lower`; P5/P6
    # patch HF classes and are only meaningful when a recipe wraps the HF
    # vision tower. All are idempotent no-ops if already applied.
    applied = apply_patches(
        # The unbacked-SymInt pair, required whenever an `.item()` call
        # reaches edge lowering — which is exactly `use_dynamic_kv_slice`.
        dynamic_kv_slice=recipe.attn.use_dynamic_kv_slice,
        # HF `Gemma4AudioModel._convert_4d_mask_to_blocked_5d` does not
        # survive a dynamic mel time dim; both multimodal recipes wrap the HF
        # audio tower, so both need it.
        dynamic_audio=recipe.has_audio,
        vision_custom_sdpa=_uses_hf_vision_tower(recipe),
        patch_embedder_where=_uses_hf_vision_tower(recipe),
    )
    _say(verbose, f"[patch] applied: {applied or '(none — already applied)'}")

    # P7 (scalar clamp) is NOT applied here: it is per-instance, so it runs
    # inside `_build_towers` once the HF modules exist, and only on the
    # HF-vision-tower path. See `patches.apply_scalar_clamp_patch`.

    # [2/8] Class flags — BEFORE construction.
    _apply_class_flags(recipe, verbose=verbose)

    # [3/8] Load + split PLE.
    model = _load_model(recipe, checkpoint, verbose=verbose)

    # [4/8] Per-attention flags — AFTER construction.
    _apply_instance_flags(model, recipe, verbose=verbose)

    # [5/8] Quantize (+ Vulkan fp16 retarget) — AFTER the PLE split.
    model = _quantize(model, recipe, verbose=verbose)

    # A 2W ring must retain W-1 previous tokens alongside the whole chunk.
    # Larger chunks evict keys still needed by the first queries in the chunk.
    if (
        recipe.attn.ring_kv_cache
        and recipe.lengths.effective_text_decoder_prefill_len > model.config.sliding_window
    ):
        raise ValueError("ring KV cache requires prefill_len <= sliding_window")

    # [6/8] Caches + source transforms — AFTER quantize.
    model = _setup_caches_and_transforms(model, recipe, verbose=verbose)

    # [7/8] Encoders. Built after the text model so the HF load, which is the
    # memory peak, does not overlap with quantisation.
    towers = _build_towers(recipe, checkpoint, verbose=verbose)

    # [8/8] Trace, lower, write.
    programs = _build_programs(model, towers, recipe, verbose=verbose)
    partitioners = _build_partitioners(recipe)
    missing = set(programs) - set(partitioners)
    if missing:
        raise RuntimeError(
            f"no partitioner for exported method(s) {sorted(missing)}"
        )
    # Narrow to the methods that actually produced a program — a partitioner
    # keyed on a method that is not in `programs` has nothing to partition.
    partitioners = {m: partitioners[m] for m in programs}
    _lower_and_write(
        programs, partitioners, recipe, out_path,
        schema_json=schema_json, verbose=verbose,
    )

    _say(verbose, f"\n[done] total wall {time.perf_counter() - t_start:.1f}s")
    return out_path
