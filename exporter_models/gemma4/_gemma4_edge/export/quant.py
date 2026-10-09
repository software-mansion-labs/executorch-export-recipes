"""Quantization recipes for the Gemma4 export pipeline.

Three pieces:

`quantize_text_model`
    Documented wrapper over `gemma4.model.text.quantize_model`, with the
    argument values the shipping recipes actually use.

`quantize_module_8da4w`
    8da4w for a vision or audio tower. One recipe, shared by the audio and
    vision paths of both multimodal exports.

`retarget_quantized_to_fp16_with_fp32_scales`
    The Vulkan fp16 bridge, plus its helper `_is_torchao_parametrized`.

`_is_torchao_parametrized` is public-ish and safe to import for the "how many
parametrized modules / what dtype are their scales" diagnostic the Vulkan
recipes print around the retarget call.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from ..model.text import quantize_model


# ---------------------------------------------------------------------------
# Text decoder — 8da4w linears + weight-only embeddings.
# ---------------------------------------------------------------------------


def quantize_text_model(
    model: nn.Module,
    *,
    linear_bits: int = 4,
    linear_group_size: int = 128,
    dynamic_activation: bool = True,
    skip_linear_name_suffixes: Sequence[str] = (),
    quantize_main_embeddings: bool = True,
    embedding_group_size: int = 32,
    quantize_ple_embeddings: bool = True,
    ple_embedding_bits: int = 4,
    ple_embedding_group_size: int = 32,
) -> nn.Module:
    """Thin, documented wrapper over `gemma4.model.text.quantize_model`.

    Quantizes in place AND returns the model. `quantize_model` ends with
    `model = unwrap_tensor_subclass(model)`, so the returned object is the
    one to keep — always write `model = quantize_text_model(model, ...)`.

    Args:
        model: `Gemma4EdgeForCausalLM`. Call `split_ple_for_export()` on
            `model.model` BEFORE this — `quantize_ple_embeddings` matches on
            the `_ple_per_layer_embs` / `_ple_per_layer_projs` FQNs that
            only exist after the split.
        linear_bits: 4 (8da4w, the ship default) or 8. Maps onto
            `quantize_model(mode=...)`.
        linear_group_size: PerGroup size over `in_features` for `nn.Linear`.
            128 in the XNNPACK multimodal recipe; the Vulkan text-only recipe
            uses 64.
        dynamic_activation: True -> `Int8DynamicActivationIntxWeightConfig`
            (8da4w), which is what lets `XnnpackDynamicallyQuantizedPartitioner`
            claim the linears. False -> weight-only `IntxWeightOnlyConfig`.
        skip_linear_name_suffixes: FQN suffixes to leave unquantized. The
            XNNPACK multimodal recipe passes
            `("q_proj", "k_proj", "v_proj", "o_proj")`: quantizing qkvo
            AND mlp together accumulated enough drift to flip the seed token
            10979 -> 9366 on a 3s-audio prompt; qkvo-only or mlp-only alone is
            fine, and dropping qkvo is also faster at decode because MLP is
            ~3x the params of QKVO. Bisected 2026-04-23.
            (`per_layer_input_gate` is excluded unconditionally inside
            `quantize_model` — its error compounds 35x across layers and
            8da4w on it alone was measured to destroy coherence.)
        quantize_main_embeddings: Weight-only quantize `embed_tokens`. When
            `config.tie_word_embeddings` is set, `quantize_model` first unties
            by materializing a real `lm_head` — otherwise one
            `dequantize_affine` would feed both the lookup and the tied linear
            and `to_executorch` rejects the dangling linear use.
        embedding_group_size: PerGroup size for `embed_tokens`. 32 in the
            XNNPACK multimodal recipe; the Vulkan text-only recipe passes its
            linear group size (64) here instead.
        quantize_ple_embeddings: Weight-only quantize the split PLE embeddings
            (and their projections, at the main weight dtype).
        ple_embedding_bits: 2, 4 or 8. 4 is the multimodal default: int2 PLE
            flipped the first-token argmax to 100 (`<|channel>`) in multimodal
            eager, and int4 recovers it — bisected 2026-04-23. The text-only
            Vulkan recipe still ships int2 (`PLE_BITS=2`), where it is
            fine. Note int2 GEMM has no kernel, so the PLE *projection*
            linears stay at the main weight dtype regardless.
        ple_embedding_group_size: PerGroup size for the PLE embeddings. 32 in
            both recipes.

    Returns:
        The quantized model (same object, mutated in place).
    """
    if linear_bits not in (4, 8):
        raise ValueError(f"linear_bits must be 4 or 8 (got {linear_bits})")
    mode = "int4" if linear_bits == 4 else "int8"

    return quantize_model(
        model,
        mode=mode,
        group_size=linear_group_size,
        quantize_main_embeddings=quantize_main_embeddings,
        embedding_group_size=embedding_group_size,
        quantize_ple_embeddings=quantize_ple_embeddings,
        ple_embedding_group_size=ple_embedding_group_size,
        ple_embedding_bits=ple_embedding_bits,
        dynamic_activation=dynamic_activation,
        # `quantize_model` gates on truthiness, so `()` reads as "skip nothing".
        skip_linear_name_suffixes=tuple(skip_linear_name_suffixes),
    )


# ---------------------------------------------------------------------------
# Vision / audio towers — 8da4w on nn.Linear only.
# ---------------------------------------------------------------------------


def quantize_module_8da4w(module: nn.Module, *, group_size: int = 128) -> nn.Module:
    """8da4w for a vision/audio tower.

    `Int8DynamicActivationIntxWeightConfig(weight_dtype=int4,
    weight_granularity=PerGroup(group_size))` applied to every `nn.Linear`
    whose `in_features` is a multiple of `group_size`, then
    `unwrap_tensor_subclass`.

    Only `nn.Linear` is touched. Conv2d weights and frontend buffers
    (hann_window / dft_real / dft_imag / mel_filters on the audio side) stay
    at their loaded dtype. The recursive `named_modules()` walk reaches the
    inner `nn.Linear` inside wrappers such as the edge tower's
    `ClippableLinear`, so the wrapper's forward keeps driving a now-quantized
    child.

    group_size=128 is the measured recipe, for both vision and audio.
    Audio quality floor vs fp32 at this setting is cos ~0.92 — the
    0.95-0.97 floor in memory is the *text* number, not this one.

    Returns the module. `unwrap_tensor_subclass` may hand back a different
    object, so always write `m = quantize_module_8da4w(m)`.
    """
    import torchao  # noqa: F401
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import (
        Int8DynamicActivationIntxWeightConfig,
        quantize_,
    )
    from torchao.utils import unwrap_tensor_subclass

    cfg = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(group_size),
    )

    def _filter(m: nn.Module, fqn: str) -> bool:
        if not isinstance(m, nn.Linear):
            return False
        in_feat = m.weight.shape[1]
        # The `>= group_size` half is implied by `% group_size == 0` for any
        # positive in_features; kept verbatim from the source recipe.
        return in_feat >= group_size and in_feat % group_size == 0

    with torch.no_grad():
        quantize_(module, cfg, filter_fn=_filter)
        # MANDATORY. Without `unwrap_tensor_subclass` the torchao tensor
        # subclasses never become real graph ops, so the XNNPACK
        # dynamically-quantized partitioner cannot see (and therefore cannot
        # claim) a single quantized linear — the whole tower falls back to
        # host execution.
        module = unwrap_tensor_subclass(module)

    return module


# ---------------------------------------------------------------------------
# Vulkan fp16 bridge.
# ---------------------------------------------------------------------------


def _is_torchao_parametrized(mod: nn.Module) -> bool:
    """A module is torchao-parametrized iff it has parametrizations.weight
    where the parametrization wraps an IntxUnpackedToInt8Tensor (signaled by
    `rebuild_stack` attribute on the first parametrization)."""
    if not hasattr(mod, "parametrizations"):
        return False
    if "weight" not in mod.parametrizations:
        return False
    pl = mod.parametrizations.weight
    if len(pl) == 0:
        return False
    return hasattr(pl[0], "rebuild_stack")


def _retarget_quantized_to_fp16_with_fp32_scales(
    model: nn.Module,
) -> Tuple[int, int, int, int]:
    """For each torchao-parametrized module:
        - Mutate rebuild_stack so reconstructed wrapper has dtype=fp16
          (this is what makes dispatch matmul run on fp16 inputs).
        - For original1 (the scale), assign upcasted-from-bf16 fp32.
        - Leave original0 (int8 qdata) and original2 (int8 zero_point) alone.

    Then cast all *other* floating-point parameters and buffers to fp16.

    Returns: (n_quantized_retargeted, n_other_params_cast, n_buffers_cast,
              n_skipped_non_floating).
    """
    n_q = 0
    n_p = 0
    n_b = 0
    n_skip = 0

    # Pass 1: walk all modules; if torchao-parametrized, retarget.
    parametrized_module_names: list[str] = []
    for name, mod in model.named_modules():
        if _is_torchao_parametrized(mod):
            parametrized_module_names.append(name)

    for fqn in parametrized_module_names:
        mod = model.get_submodule(fqn)
        pl = mod.parametrizations.weight
        ut = pl[0]
        # Mutate rebuild_stack: replace dtype=bf16 with dtype=fp16 in every
        # frame of the stack (there's typically one frame for our setup).
        new_stack = []
        for tp, meta, inner in ut.rebuild_stack:
            new_meta = dict(meta)
            old_dtype = new_meta.get("dtype")
            if old_dtype in (torch.bfloat16, torch.float32):
                new_meta["dtype"] = torch.float16
            new_stack.append((tp, new_meta, inner))
        ut.rebuild_stack = new_stack

        # Upcast scale (original1) bf16 → fp32 (lossless: bf16 mantissa fits
        # in fp32 mantissa exactly).
        old_scale = pl.original1.data
        # `.float()` upcasts losslessly from bf16 (or fp16, or stays at fp32).
        pl.original1.data = old_scale.float()
        n_q += 1

    # Pass 2: cast every NON-quantized floating param/buffer to fp16. Need to
    # be careful not to touch parametrization-internal originals — those carry
    # int8 qdata (we want to keep them int8) and the fp32 scale we just set.
    floating_dtypes = (torch.float32, torch.bfloat16, torch.float64)

    # Collect all parametrization-internal parameters to skip.
    parametrization_params: set[str] = set()
    for fqn in parametrized_module_names:
        for sub_name, _ in (
            model.get_submodule(fqn).parametrizations.weight.named_parameters()
        ):
            parametrization_params.add(f"{fqn}.parametrizations.weight.{sub_name}")

    for name, p in model.named_parameters():
        if name in parametrization_params:
            continue
        if p.dtype in floating_dtypes:
            p.data = p.data.to(torch.float16)
            n_p += 1
        elif p.dtype == torch.float16:
            pass
        else:
            n_skip += 1

    for name, b in model.named_buffers():
        if b.dtype in floating_dtypes:
            # In-place reassign in the parent module's buffer registry.
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            new_buf = b.to(torch.float16)
            try:
                if attr_name in parent._buffers:
                    parent._buffers[attr_name] = new_buf
                else:
                    setattr(parent, attr_name, new_buf)
            except Exception:
                setattr(parent, attr_name, new_buf)
            n_b += 1
        elif b.dtype == torch.float16:
            pass
        else:
            n_skip += 1

    return n_q, n_p, n_b, n_skip


def retarget_quantized_to_fp16_with_fp32_scales(model: nn.Module) -> None:
    """The Vulkan fp16 bridge.

    Run AFTER `quantize_text_model` / `quantize_module_8da4w` on a model that
    was loaded at bf16. It rewrites each torchao `rebuild_stack` frame's dtype
    to fp16 — so the reconstructed wrapper dispatches its matmul on fp16
    inputs, which is what the Vulkan `linear_q4gsw_*` shaders want — while
    upcasting `original1` (the scale) to fp32 and leaving `original0` (int8
    qdata) and `original2` (int8 zero_point) untouched. Everything else that
    is still floating (RMSNorm weights, plain embeddings, RoPE buffers,
    biases) is cast to fp16.

    Why fp32 scales: bf16 -> fp16 rounding of the per-group scales was the
    dominant fp16 accuracy floor. bf16 -> fp32 is lossless (the bf16 mantissa
    fits exactly), so keeping the scale wide and the activations narrow buys
    the precision back for free. Paired with the fp32-scale shaders and a
    relinked runner, this took the on-device text decoder to cos = 0.99 vs
    HF bf16.

    Ordering constraint: this MUST run after quantization (there are no
    `rebuild_stack` frames to rewrite before it) and BEFORE `setup_caches(...,
    dtype=torch.float16)` and the source transforms.

    Mutates `model` in place and returns None. The private
    `_retarget_quantized_to_fp16_with_fp32_scales` returns the
    (n_quantized, n_params_cast, n_buffers_cast, n_skipped) counts that the
    Vulkan recipes print, if a caller wants them for a diagnostic.
    """
    _retarget_quantized_to_fp16_with_fp32_scales(model)
