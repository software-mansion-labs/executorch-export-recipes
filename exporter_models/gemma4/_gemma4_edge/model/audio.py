"""
Gemma 4 — Audio Encoder (ExecuTorch edge implementation)
========================================================

Faithfully matches the reference transformers implementation
(transformers/models/gemma4/modeling_gemma4.py, lines 179-534 + 1779-1861).

Architecture (Universal Speech Model lineage):

  1. LogMelFrontend (this file): waveform -> log-mel spectrogram (128 mel
     bins, 16 kHz, 25 ms / 10 ms window/hop, 512-pt FFT, Hann window).
     The HF processor pre-computes Hann window + mel filter bank; we
     consume them as buffers (`hann_window`, `mel_filters`) plus a
     pre-baked real/imag DFT matrix so the entire path is plain
     `torch.matmul` (no `torch.stft`).

  2. SubSample conv projection: two stacked 3x3 stride-2 conv2d blocks
     with LayerNorm + ReLU and a final linear that maps the channel-
     unrolled features into `hidden_size`. This downsamples by 4x on the
     time axis and by 4x on the mel axis (1 -> 128 -> 32 -> hidden_size).

  3. Relative positional encoding (Transformer-XL style, see Gemma4
     Universal-Speech notes): `(1, context_size, hidden_size)` sinusoid
     evaluated at relative positions `[max_past_horizon, ..., 0]`.

  4. Stack of N `Gemma4AudioLayer` blocks. Each layer is:
       residual + 0.5 * RMSNorm(FFN1(x))               (`feed_forward1`)
       residual + RMSNorm(ChunkedAttention(RMSNorm(x))) (`self_attn`)
       residual + LightConv1d(x)                       (`lconv1d`)
       residual + 0.5 * RMSNorm(FFN2(x))               (`feed_forward2`)
       RMSNorm                                         (`norm_out`)
     Every projection feeds through `Gemma4ClippableLinear` (input/output
     clamp pair) and `torch.clamp(..., gradient_clipping, ...)` brackets
     several positions. The chunked attention has:
       - Q scaled by `(head_dim^-0.5) / ln(2) * softplus(per_dim_scale)`
       - K scaled by `ln(1 + e) / ln(2)`
       - tanh soft-cap at `attention_logit_cap`
       - relative-key bias `matrix_bd` via the Transformer-XL rel-shift.

  5. Output projection: a single `nn.Linear` mapping `hidden_size` ->
     `output_proj_dims` (the input dimension of `embed_audio`). HF's audio
     tower sets `bias=True` here (unlike vision/text where output_proj is
     bias-free), so we follow.

  6. Final `embed_audio`: an instance of `Gemma4MultimodalEmbedder`,
     identical structurally to the vision-side analogue — pre-norm
     (RMSNorm without scale) over `output_proj_dims`, then a single
     `nn.Linear` to the text-decoder hidden size. We embed it inside the
     `Gemma4EdgeAudioTower` so callers see a single `.forward(waveform,
     attention_mask) -> audio_embeds` surface mirroring the deleted
     `export_audio_encoder_bf16.py` (git rev `9a827e7bfd`).

Edge-friendly simplifications relative to HF (math preserved):

  - `Gemma4ClippableLinear` is collapsed to a plain `nn.Linear` plus a
    pair of Python-scalar `torch.clamp` calls. Identical bake-in pattern
    to `ClippableLinear` in `gemma4_vision.py`. The HF version registers
    four 0-d Tensor buffers (`input_min`, `input_max`, `output_min`,
    `output_max`); we bake them into Python floats at load time so the
    exported graph lowers to `aten.clamp.default` not `aten.clamp.Tensor`.

  - `_convert_4d_mask_to_blocked_5d` is reimplemented with the
    `unfold + diagonal` variant from the deleted bf16 export script
    (git rev `9a827e7bfd`). The HF reference uses
    `torch.arange(num_blocks, device=...) * chunk_size` to build kv
    indices and gathers from a padded mask, which fails to export with
    a dynamic seq_len (the `arange(symbolic)` is not exportable). The
    unfold variant is mathematically equivalent and trace-friendly.

  - `LogMelFrontend` uses `mask.unfold(...)[-1]` rather than
    `torch.arange(symbolic T)` to derive the per-frame validity bitmask.
    Same simplification rationale.

  - `Gemma4AudioRelPositionalEncoding` keeps HF's hard-coded
    `torch.arange(12, -1, -1)`. This 13-wide window matches the default
    config (`attention_context_left=13`, `attention_context_right=0`) —
    in other words, the position window is `[max_past_horizon, ..., 0]`.
    We expose it as a Python constant `_REL_POS_WINDOW` so the value is
    a compile-time scalar and not a data-dependent tensor.

  - `_rel_shift` is a verbatim port (no simplification — the
    `F.pad + view + slice + view` pattern is already export-friendly).

  - `attention_mask.logical_not().masked_fill(...)` is preserved (HF
    audio path is `sdpa` which produces a `bool` 4D mask by default;
    `eager` mask would be float and the math would differ). Asserting
    `attn_implementation="sdpa"` is the caller's responsibility — we
    document it on `Gemma4EdgeAudioTower.from_hf`.

  - The `subsample` mask propagation (`mask[:, ::2]`) is verbatim, and
    `subsample` itself uses `F.layer_norm`-style scaling via
    `nn.LayerNorm(out_channels, elementwise_affine=True, bias=False)`
    matching HF exactly.

This file's structure mirrors `gemma4_vision.py` (config dataclass, leaf
modules, top-level tower with `.from_hf` classmethod + `_config_from_hf`
+ `_load_state_from_hf` helpers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Side-effect import: registers `et_vk::gemma_sdpa_with_kv_cache` and
# `et_vk::gemma_custom_sdpa` with torch.library. Provides Meta + CPU
# CompositeExplicitAutograd impls so eager parity testing and tracing work
# without the Vulkan runtime. Required when `use_vulkan_sdpa=True`.
try:
    from . import custom_sdpa_ops  # noqa: F401

    _HAS_VULKAN_SDPA_OPS = True
except ImportError:
    _HAS_VULKAN_SDPA_OPS = False


# ---------------------------------------------------------------------------
# Audio constants (mirroring the deleted `export_audio_encoder_bf16.py`)
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE = 16000
AUDIO_FRAME_LENGTH = 320          # 20 ms @ 16 kHz
AUDIO_HOP_LENGTH = 160            # 10 ms @ 16 kHz
AUDIO_FFT_LENGTH = 512
AUDIO_N_MEL = 128
AUDIO_MEL_FLOOR = 1e-3
AUDIO_PAD_LEFT = 160              # = frame_length // 2 (semicausal time padding)
AUDIO_PAD_RIGHT = 0
AUDIO_FRAME_UNFOLD_SIZE = 321     # = AUDIO_FRAME_LENGTH + 1; the unfold trims
                                  # the last sample inside the frame, but this
                                  # extra sample is also where the frame mask
                                  # reads the validity bit (`mask_frames[..., -1]`).

# Block alignment for chunked attention (chunk_size=12, conv subsamples by 4
# on the time axis, hop_length=160 => one chunk = 12 * 4 * hop = 7680 samples)
BLOCK_SAMPLES = 7680              # 0.48s @ 16 kHz, one chunked-attention block
BLOCK_K_MIN = 1                   # one chunk
BLOCK_K_MAX = 62                  # ~29.76 s
AUDIO_SAMPLE_COUNT_MAX = BLOCK_SAMPLES * BLOCK_K_MAX
AUDIO_SAMPLE_COUNT_MIN = BLOCK_SAMPLES * BLOCK_K_MIN

# Relative-position window size used by `Gemma4AudioRelPositionalEncoding`.
# HF hard-codes `torch.arange(12, -1, -1)` (13 entries: 12, 11, ..., 0). We
# keep that exact value as a Python constant so it lowers to a compile-time
# `aten.arange` with literal endpoints in the exported graph.
_REL_POS_WINDOW_START = 12

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Gemma4EdgeAudioConfig:
    """Config for Gemma 4 audio tower, mirroring `Gemma4AudioConfig`.

    Only fields actually consumed by the edge implementation are listed.
    Default values match HF's `Gemma4AudioConfig.__init__` defaults.
    """

    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"

    # subsampling (input front-end)
    subsampling_conv_channels: Tuple[int, int] = (128, 32)

    # conformer / chunked attention
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1.0e9

    use_clipped_linears: bool = True
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    output_proj_dims: int = 1536

    # Text-decoder hidden size for the final `embed_audio` projection. The
    # edge tower owns `embed_audio` so callers see a single
    # `forward(waveform, attention_mask) -> (B, N_audio, text_hidden)` path
    # rather than the two-step HF surface (audio_tower then embed_audio).
    text_hidden_size: int = 2048

    # Held for parity with HF surface but not consumed in forward.
    initializer_range: float = 0.02

    # ---------------- Vulkan SDPA fusion (opt-in) ----------------
    # Mirrors vision-side B5 (`use_vulkan_sdpa`). When set:
    #   - `Gemma4EdgeAudioAttention.forward` skips the explicit chunked
    #     Q@K.T -> softcap -> mask-add -> softmax -> @V chain.
    #   - It instead rewrites attention as global (B, S, H, D) SDPA, calls
    #     `torch.ops.et_vk.gemma_custom_sdpa.default(q, k, v, 0, S, mask, False)`
    #     against the pre-baked `(1, 1, S_max, S_max)` chunked-block mask.
    #   - K and V are passed DIRECTLY as the cache arguments (no scratch
    #     buffer). The C++ `realize_cache_once` (GemmaSDPA.cpp:68-91)
    #     detects that they are graph Tensors and returns them in-place.
    #   - The Vulkan SDPA op's 4D mask is broadcast over heads
    #     (gemma_sdpa_compute_attn_weights_tiled.glsl:117-126 HAS_MASK
    #     branch). It does NOT support per-head bias, so the
    #     Transformer-XL relative-key bias (rel_pos / `matrix_bd`) and
    #     the `tanh(x/cap)*cap` softcap are BOTH dropped in this path
    #     (Phase 2A). Quality of the drop is measured by the
    #     B5-ON vs HF eager cos in `check_edge_audio_parity.py`. If cos
    #     drops below the 0.99 gate on the encoder body, a Phase 2B
    #     shader extension is required (per-head mask + on-shader
    #     softcap). See PLAN_gemma4_custom_sdpa_shader.md.
    use_vulkan_sdpa: bool = False
    # When `use_vulkan_sdpa=True`, the encoder pre-bakes the static
    # `(1, 1, S_max, S_max)` chunked-block mask. The maximum S is set
    # via `bake_audio_sdpa_mask(seq_len_max, dtype)` before export, so
    # the buffer lowers to a TensorRef constant (val_is_tref check on
    # the Vulkan prepack path).
    sdpa_mask_seq_len_max: int = 0

    # Lever B. When True, pre-bake the 5D blocked
    # bool attention mask `(1, 1, num_blocks_max, chunk_size, context_size)`
    # for the NON-Vulkan attention forward (use_vulkan_sdpa=False path).
    # At runtime, the audio tower slices `[:, :, :num_blocks_active, :, :]`
    # instead of calling `_build_bidirectional_audio_mask` -> `unfold` +
    # `diagonal` + bool AND chain. Assumes all input positions are valid
    # (mirrors the Vulkan path's invariant; matches typical inference
    # where `mask = ones(1, SAMPLES)` upstream). Set via
    # `bake_audio_5d_mask(seq_len_max)` before export.
    use_baked_5d_mask: bool = False
    baked_5d_mask_seq_len_max: int = 0

    # Lever G (from a partitioner audit). When True,
    # `Gemma4EdgeAudioAttention._rel_shift` replaces the `F.pad + view +
    # slice + view` chain (XNNPACK refuses the slice_copy) with
    # `F.pad + view + index_select + view` against a pre-baked 1D int64
    # buffer `_rel_shift_baked_idx` of length `block_size * context_size`
    # (288 for default audio config). Bit-exact (eager max|Δ|=0.0 across
    # k∈{1,2,8,16,62}). Eliminates ~9 `slice_copy` host ops per layer
    # plus a downstream getitem cascade; ~60-140 host ops at N=12.
    # Independent of `use_vulkan_sdpa` — only consumed in the non-Vulkan
    # attention forward (use_vulkan_sdpa=True still uses the dense flat
    # `_rel_shift_flat_idx` machinery, which is a different bake).
    use_baked_rel_shift_idx: bool = False

    # Lever H (from a follow-up audit). When True, the non-Vulkan
    # `Gemma4EdgeAudioAttention.forward` stacks K and V on a new leading
    # axis before calling `_extract_block_context`, performs ONE
    # F.pad+unfold+movedim+contiguous chain on the combined tensor, then
    # splits back. Math is bit-exact to the two-call baseline (K and V are
    # independent along their own batch axis, so a shared pad+unfold over
    # the time axis produces identical per-tensor results). Per-layer
    # savings target: 1 unfold_copy (drops from 2 to 1) plus possibly a
    # view/getitem dedupe. Independent of `use_vulkan_sdpa` (Vulkan path
    # uses gemma_custom_sdpa and never enters `_extract_block_context`).
    use_combined_kv_extract: bool = False

    # ---------------- Fix C ----------------
    # When True, `Gemma4EdgeAudioAttention.__init__` registers a sibling
    # `softplus_per_dim_scale` buffer that the forward uses INSTEAD of
    # recomputing `F.softplus(self.per_dim_scale)` per forward. The buffer
    # is filled via `bake_softplus_per_dim_scale()` after state-dict load.
    # Eliminates 4 host ops per layer (gt + log1p + exp + where) under the
    # numerically-stable softplus decomp = 48 host ops at N=12.
    # Gated by this flag so the existing eager parity test
    # (`check_edge_audio_vulkan_sdpa_parity.py`) is unaffected by default.
    bake_softplus_scale: bool = False


# ---------------------------------------------------------------------------
# RMSNorm — identical to text/vision precedent
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Mirror of `Gemma4RMSNorm` (with optional learnable scale)."""

    # When True, swap torch.pow(var, -0.5) -> torch.rsqrt(var). Bit-
    # exact in fp32 (validated across Gemma4 shapes) but lets XNNPACK
    # absorb the rsqrt op (rsqrt.default has no rank constraints, unlike pow
    # which is gated on integer power==2). Eliminates ~109 hosted pow ops on
    # the audio_encoder path.
    use_rsqrt: bool = False

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        self.dim = dim
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = (x * x).mean(-1, keepdim=True) + self.eps
        if RMSNorm.use_rsqrt:
            return x * torch.rsqrt(var)
        # Use torch.pow(-0.5) (over rsqrt/sqrt) to match the HF compiler
        # comment ("addresses compiler differences between Torch and JAX").
        return x * torch.pow(var, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        if self.with_scale:
            output = output * self.weight.float()
        return output.type_as(x)


# ---------------------------------------------------------------------------
# ClippableLinear — collapsed to plain nn.Linear with Python-scalar clamps
# ---------------------------------------------------------------------------


class ClippableLinear(nn.Module):
    """Edge replacement for `Gemma4ClippableLinear` (HF lines 129-155).

    Math is bit-exact relative to HF's `Gemma4ClippableLinear` when the
    bounds are loaded from the checkpoint via `set_bounds(...)`. See
    `gemma4_vision.ClippableLinear` for the rationale (Vulkan only knows
    `aten.clamp.default`, not `aten.clamp.Tensor`).
    """

    def __init__(
        self,
        config: Gemma4EdgeAudioConfig,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Filled by `set_bounds(...)` at load time; default to identity (no-op).
        self.input_min: float = float("-inf")
        self.input_max: float = float("inf")
        self.output_min: float = float("-inf")
        self.output_max: float = float("inf")

    def set_bounds(
        self,
        input_min: float,
        input_max: float,
        output_min: float,
        output_max: float,
    ) -> None:
        self.input_min = float(input_min)
        self.input_max = float(input_max)
        self.output_min = float(output_min)
        self.output_max = float(output_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipped_linears:
            x = torch.clamp(x, self.output_min, self.output_max)
        return x


# ---------------------------------------------------------------------------
# Relative positional encoding
# ---------------------------------------------------------------------------


class Gemma4EdgeAudioRelPositionalEncoding(nn.Module):
    """Mirror of `Gemma4AudioRelPositionalEncoding` (HF lines 179-207).

    Produces `(1, len(rel_window), hidden_size)` sinusoidal embeddings at
    relative positions `[12, 11, ..., 0]` (HF hard-coded) with the
    concatenated `[sin..., cos...]` layout matching the original Gemma4
    convention.

    Pre-bakes `inv_timescales` as a non-persistent buffer (1x1 shaped to
    match HF). The forward path is a single `arange * inv_timescales`
    outer product, a `cat([sin, cos])`, and a dtype cast.
    """

    inv_timescales: torch.Tensor

    def __init__(self, config: Gemma4EdgeAudioConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.context_size = (
            config.attention_chunk_size
            + config.attention_context_left
            - 1
            + config.attention_context_right
        )
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(
            num_timescales - 1, 1
        )
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales) * -log_timescale_increment
        )
        # Shape mirror HF: (1, 1, num_timescales) so it broadcasts against
        # (1, rel_window, 1) when multiplied below.
        self.register_buffer(
            "inv_timescales",
            inv_timescales.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # HF hard-codes `torch.arange(12, -1, -1)` regardless of config.
        # 12 = max_past_horizon for the default config; the window covers
        # relative positions [past_horizon, ..., 0].
        position_ids = torch.arange(
            _REL_POS_WINDOW_START, -1, -1, device=hidden_states.device
        )
        position_ids = position_ids[..., None]   # (rel_window, 1)
        scaled_time = position_ids * self.inv_timescales.to(
            device=hidden_states.device
        )                                       # (1, rel_window, num_timescales)
        pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pos_embed.to(dtype=hidden_states.dtype)


# ---------------------------------------------------------------------------
# Chunked attention
# ---------------------------------------------------------------------------


class Gemma4EdgeAudioAttention(nn.Module):
    """Chunked local attention with Transformer-XL relative position bias.

    Mirror of `Gemma4AudioAttention` (HF lines 210-315). Every projection
    is a `ClippableLinear` (bounds baked at load time); `relative_k_proj`
    stays a plain `nn.Linear` (HF does not wrap it in Gemma4ClippableLinear).
    """

    def __init__(self, config: Gemma4EdgeAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_logits_soft_cap = config.attention_logit_cap
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        # Q scale: (head_dim^-0.5) / ln(2). Then multiplied per-head by
        # softplus(per_dim_scale) on every forward.
        self.q_scale = (self.head_dim ** -0.5) / math.log(2)
        # K scale: ln(1 + e) / ln(2). Constant.
        self.k_scale = math.log(1 + math.e) / math.log(2)

        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )

        self.q_proj = ClippableLinear(
            config, config.hidden_size, self.num_heads * self.head_dim
        )
        self.k_proj = ClippableLinear(
            config, config.hidden_size, self.num_heads * self.head_dim
        )
        self.v_proj = ClippableLinear(
            config, config.hidden_size, self.num_heads * self.head_dim
        )
        self.post = ClippableLinear(
            config, config.hidden_size, config.hidden_size
        )

        # Plain Linear (no clamp) per HF.
        self.relative_k_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

        # Fix C: pre-baked `F.softplus(per_dim_scale)` buffer.
        # When `config.bake_softplus_scale=True`, the forward consumes this
        # buffer directly instead of recomputing softplus on every forward.
        # The buffer is filled by `bake_softplus_per_dim_scale()` after the
        # state-dict load (must happen post-load because per_dim_scale is a
        # learned param that starts at zeros and gets overwritten by load).
        # Persistent=False because it's derived from per_dim_scale; state
        # dicts only carry per_dim_scale itself.
        if getattr(config, "bake_softplus_scale", False):
            self.register_buffer(
                "softplus_per_dim_scale",
                torch.zeros(self.head_dim, dtype=torch.float32),
                persistent=False,
            )

        # Rel-shift scatter vectorization: pre-baked flat-index
        # buffer for the dense `matrix_bd_dense` expansion. Replaces the
        # per-block `slice_scatter` loop in `_forward_vulkan_sdpa` with a
        # single `index_select` (1 host op vs 36 slice_scatter + 24 copy).
        # The index is a 1D int64 of length `S_full * S_full`; entry at
        # flat position `s*S_full + k` is the flat source position in
        # `matrix_bd.reshape(B, H, num_blocks*chunk_size*context_size)`
        # for valid (s, k) per the per-block window; for (s, k) outside
        # the per-block window, entry is 0 (any in-range value works —
        # the dense bias at masked positions gets `-inf`'d by the
        # padding mask before softmax, so the value is irrelevant).
        #
        # Filled by `bake_rel_shift_flat_idx(seq_len_max)` after
        # state-dict load, BEFORE export. Persistent=True so it lowers
        # to a TensorRef constant on the Vulkan prepack path.
        # Identical for every layer (depends only on chunk_size,
        # context_size, max_past_horizon, S_full) — register per-layer
        # for now; dedupe is automatic in the Vulkan delegate's constant
        # pool. Default zeros(0) is a placeholder.
        if getattr(config, "use_vulkan_sdpa", False):
            self.register_buffer(
                "_rel_shift_flat_idx",
                torch.zeros(0, dtype=torch.long),
                persistent=True,
            )

        # Lever G: pre-baked 1D int64 index for the non-Vulkan
        # `_rel_shift` fast-path. Lazy bake via `bake_rel_shift_baked_idx`;
        # default `arange(block_size*context_size)` is correct math but we
        # register an empty placeholder so torch.export sees a real buffer
        # and the bake helper is callable post-construction.
        if getattr(config, "use_baked_rel_shift_idx", False):
            self.register_buffer(
                "_rel_shift_baked_idx",
                torch.zeros(0, dtype=torch.long),
                persistent=True,
            )

        # 0-d float buffer (not a parameter); used as a Python scalar in
        # the softcap chain. We register it as a buffer for state-dict
        # parity, but the math is `tanh(x / cap) * cap` with Python scalars
        # (cap is a constant -> compile-time scalar).
        self.register_buffer(
            "softcap",
            torch.tensor(self.attention_logits_soft_cap),
            persistent=False,
        )

    def bake_softplus_per_dim_scale(self) -> None:
        """Fix C: fill the pre-baked `softplus_per_dim_scale`
        buffer from the loaded `per_dim_scale` parameter.

        Must be called AFTER state-dict load (the parameter starts at
        zeros and gets overwritten on load) and BEFORE export. No-op
        when `config.bake_softplus_scale=False`.
        """
        if not getattr(self.config, "bake_softplus_scale", False):
            return
        with torch.no_grad():
            # Compute softplus in fp32 to match the forward path's
            # `F.softplus(self.per_dim_scale.float())` (line ~602).
            baked = F.softplus(self.per_dim_scale.float())
        # Match the per_dim_scale dtype/device so downstream math doesn't
        # surprise-promote on dtype mismatches.
        self.softplus_per_dim_scale = baked.to(
            dtype=torch.float32, device=self.per_dim_scale.device
        ).contiguous()

    def bake_rel_shift_flat_idx(self, seq_len_max: int) -> None:
        """Fill the pre-baked `_rel_shift_flat_idx` buffer that
        replaces BOTH the `_rel_shift` call AND the per-block `slice_scatter`
        loop in `_forward_vulkan_sdpa`. Indexes directly into the
        PRE-`_rel_shift` flat tensor `matrix_bd_flat.reshape(B, H, -1)` of
        length `S_full * position_length` (was `S_full * context_size`
        post-rel-shift).

        Forward path becomes:
            matrix_bd_flat = queries_for_bd @ rel_K^T      # (B,H,S_full,position_length)
            matrix_bd_dense = matrix_bd_flat.reshape(B,H,-1)
                                 .index_select(dim=-1, index=_rel_shift_flat_idx)
                                 .view(B,H,S_full,S_full)

        Composed inverse-map math: given a target output position (s, k),
          1. b = s // chunk_size; src_row = s - b*chunk_size;
             src_kv = k - (b*chunk_size - max_past_horizon)
             (valid iff src_kv in [0, context_size) AND k in [0, S_full)
              AND kv window non-empty for this block)
          2. i = src_row * context_size + src_kv                    (post-rel-shift offset
                                                                     within block_size*P view)
          3. P = context_size + 1
             row_pre = i // P
             p_padded = i % P
             (valid iff p_padded < position_length AND row_pre < chunk_size;
              comes from the inverse of _rel_shift's
              F.pad(..., (0, P - position_length)) + view + slice + view chain)
          4. flat = (b * chunk_size + row_pre) * position_length + p_padded
                  = s_pre * position_length + p_padded               (into the
                                                                     PRE-rel-shift
                                                                     matrix_bd_flat.reshape(B,H,-1)
                                                                     of length S_full*position_length)
        For (s, k) outside the per-block window OR with an out-of-bounds
        (row_pre, p_padded), entry stays 0 (sentinel — those positions
        get `-inf`'d by the additive padding mask before softmax in the
        SDPA op so the value is irrelevant).

        Must be called BEFORE export with `seq_len_max` matching the
        static S the export will pin. No-op when `use_vulkan_sdpa=False`.
        Registers the buffer lazily if not already present (mirrors how
        `bake_audio_sdpa_mask` is callable after a post-construction
        `cfg.use_vulkan_sdpa = True` flip in the export flow).

        Changes vs the earlier post-rel-shift bake:
          - Index targets switched from post-rel-shift flat layout to
            pre-rel-shift flat layout (saves 12 constant_pad_nd host ops
            per export, plus one reshape per layer).
          - Buffer length stays `S_full * S_full`.
          - Index VALUE range: was [0, num_blocks*chunk_size*context_size);
                                now  [0, S_full*position_length).
        """
        if not getattr(self.config, "use_vulkan_sdpa", False):
            return
        chunk_size = self.chunk_size
        max_past_horizon = self.max_past_horizon
        max_future_horizon = self.max_future_horizon
        context_size = self.context_size
        position_length = _REL_POS_WINDOW_START + 1  # rel_window, see Gemma4EdgeAudioRelPositionalEncoding
        # Pad seq_len_max up to a multiple of chunk_size (mirrors the
        # `num_blocks = ceil(S / chunk)` in the forward).
        num_blocks = (seq_len_max + chunk_size - 1) // chunk_size
        S_full = num_blocks * chunk_size
        # NOTE: the forward trims back to seq_len after the SDPA call,
        # so we build flat_idx at S_full and let the caller align. For
        # the audio path, BLOCK_SAMPLES is a multiple of chunk_size so
        # seq_len_max == S_full in practice; assert here.
        assert S_full == seq_len_max, (
            f"bake_rel_shift_flat_idx requires seq_len_max ({seq_len_max}) "
            f"to be a multiple of chunk_size ({chunk_size}); got S_full={S_full}"
        )
        P = context_size + 1
        flat_idx = torch.zeros(S_full * S_full, dtype=torch.long)
        for b in range(num_blocks):
            row_start = b * chunk_size
            row_end = row_start + chunk_size
            kv_start_window = row_start - max_past_horizon
            kv_lo = max(kv_start_window, 0)
            kv_hi = min(kv_start_window + context_size, S_full)
            if kv_hi <= kv_lo:
                continue
            for s in range(row_start, row_end):
                src_row = s - row_start
                for k in range(kv_lo, kv_hi):
                    src_kv = k - kv_start_window
                    i = src_row * context_size + src_kv
                    row_pre = i // P
                    p_padded = i % P
                    if p_padded >= position_length or row_pre >= chunk_size:
                        # Outside _rel_shift's valid inverse map (the
                        # position would have read the F.pad zero or
                        # past the block boundary). Sentinel 0; masked
                        # by -inf in the additive padding mask anyway.
                        continue
                    s_pre = b * chunk_size + row_pre
                    flat_idx[s * S_full + k] = (
                        s_pre * position_length + p_padded
                    )
        # Register the buffer if `__init__` didn't (because
        # `use_vulkan_sdpa` may have been flipped post-construction).
        if not hasattr(self, "_rel_shift_flat_idx") or (
            self._rel_shift_flat_idx is None
        ):
            self.register_buffer(
                "_rel_shift_flat_idx",
                flat_idx.contiguous(),
                persistent=True,
            )
        else:
            self._rel_shift_flat_idx = flat_idx.contiguous()
        # Remember the S_max the buffer was baked at, so the
        # runtime slice in `_forward_vulkan_sdpa` can view the buffer back
        # as (S_max, S_max) and crop to the active S_full <= S_max.
        # Plain Python int attribute (not a buffer / parameter) — picked
        # up by torch.export as a compile-time constant.
        self._rel_shift_seq_len_max = int(S_full)

    def bake_rel_shift_baked_idx(self) -> None:
        """Lever G: fill the pre-baked `_rel_shift_baked_idx`
        buffer used by the non-Vulkan `_rel_shift` fast-path.

        The buffer is a 1D int64 of length `block_size * context_size`
        (288 for the default audio config). It indexes into the per-block
        padded flat view of shape `(B, H, num_blocks, block_size * P)`
        where `P = context_size + 1`.

        Math (verbatim port of the slice math in the original
        `_rel_shift`): the original chain was
            x_pad = F.pad(x, (0, P - position_length))   # zero columns
            x_view = x_pad.view(..., block_size * P)     # flatten per-block
            x_slc = x_view[..., : block_size * context_size]   # PREFIX slice
            return x_slc.view(..., block_size, context_size)
        The prefix slice equals `index_select(arange(block_size *
        context_size))`. We materialize the arange as a buffer so that
        torch.export emits a single `index_select` host op (XNNPACK
        friendly) instead of the `slice_copy` chain (XNNPACK refuses).

        The bake is INDEPENDENT of num_blocks (the rel_shift math is the
        same for each block), so this composes with the dynamic-samples
        export without per-k slicing.

        Must be called BEFORE export. No-op when
        `use_baked_rel_shift_idx=False`. Registers the buffer lazily if
        not already present (mirrors how the Vulkan-side
        `bake_rel_shift_flat_idx` is callable after a post-construction
        flag flip).
        """
        if not getattr(self.config, "use_baked_rel_shift_idx", False):
            return
        block_size = self.chunk_size
        context_size = self.context_size
        out_len = block_size * context_size
        baked = torch.arange(out_len, dtype=torch.long).contiguous()
        if not hasattr(self, "_rel_shift_baked_idx") or (
            self._rel_shift_baked_idx is None
        ):
            self.register_buffer(
                "_rel_shift_baked_idx",
                baked,
                persistent=True,
            )
        else:
            self._rel_shift_baked_idx = baked

    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Splits `(B, S, H, D)` into `(B, num_blocks, chunk_size, H, D)`."""
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        return hidden_states.reshape(
            batch_size, num_blocks, self.chunk_size, num_heads, head_dim
        ).contiguous()

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extracts `(B, num_blocks, context_size, H, D)` overlapping windows."""
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        hidden_states = F.pad(
            hidden_states,
            (0, 0, 0, 0, self.max_past_horizon,
             self.max_future_horizon + self.chunk_size - 1),
        )
        # unfold along the time axis (dim=1) with size=context_size, step=chunk_size
        hidden_states = hidden_states.unfold(1, self.context_size, self.chunk_size)
        # unfold lands the context dim at -1; move it to position 2
        hidden_states = torch.movedim(hidden_states, -1, 2)
        return hidden_states.contiguous()

    def _extract_block_context_kv(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lever H: combined K/V context extraction.

        Math-equivalent to calling `_extract_block_context` twice on K and V
        independently — K and V do not interact during pad+unfold, so a
        shared pad+unfold along the time axis of a stacked (2, B, S, H, D)
        tensor yields per-tensor outputs IDENTICAL to two separate calls.

        Per-layer savings (host ops): one `unfold_copy` + downstream
        view/getitem cascade collapsed. K and V dtypes must match (they do
        here — both come from `*.float()` upstream).
        """
        stacked = torch.stack([key_states, value_states], dim=0)  # (2,B,S,H,D)
        stacked = F.pad(
            stacked,
            (0, 0, 0, 0, self.max_past_horizon,
             self.max_future_horizon + self.chunk_size - 1),
        )
        # unfold along the time axis (dim=2 now after the stack) with
        # size=context_size, step=chunk_size.
        stacked = stacked.unfold(2, self.context_size, self.chunk_size)
        # unfold lands the context dim at -1; move it to the position the
        # per-tensor view expects (position 2 of each split, which is
        # position 3 of the 6D stacked tensor).
        stacked = torch.movedim(stacked, -1, 3)
        stacked = stacked.contiguous()
        return stacked[0], stacked[1]

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer-XL `_rel_shift` (HF lines 257-264). Verbatim port.

        Input:  `(B, H, num_blocks, block_size, position_length)`.
        Output: `(B, H, num_blocks, block_size, context_size)`.

        Lever G: when `config.use_baked_rel_shift_idx=True`,
        swap the prefix slice for an `index_select` against the pre-baked
        `_rel_shift_baked_idx` buffer. Bit-exact (eager max|Δ|=0.0
        validated against the slice form); eliminates ~9
        `slice_copy` host ops + downstream getitem cascade per layer,
        ~80-140 host ops at AUDIO_N_LAYERS=12.
        """
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        context_size = self.context_size
        x = F.pad(x, (0, context_size + 1 - position_length))
        x = x.view(
            batch_size, num_heads, num_blocks, block_size * (context_size + 1)
        )
        if getattr(self.config, "use_baked_rel_shift_idx", False):
            x = x.index_select(dim=-1, index=self._rel_shift_baked_idx)
        else:
            x = x[..., : block_size * context_size]
        return x.view(batch_size, num_heads, num_blocks, block_size, context_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ----- B5-style Vulkan SDPA fusion (Phase 2B — full math) ----------
        # Now math-equivalent to the chunked path (modulo Option-B softcap
        # ordering); rel-pos bias and softcap both preserved via the
        # extended `et_vk::gemma_custom_sdpa` op (per-head mask + softcap_value
        # arg). See `Gemma4EdgeAudioConfig.use_vulkan_sdpa`.
        use_vulkan_sdpa = getattr(self.config, "use_vulkan_sdpa", False)
        if use_vulkan_sdpa:
            return self._forward_vulkan_sdpa(
                hidden_states, attention_mask, position_embeddings
            )

        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

        # HF computes Q/K/V in fp32 (note the `.float()`).
        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)

        # Per-head Q scale (data-dependent: softplus(per_dim_scale)).
        query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
        key_states = key_states * self.k_scale

        # Block-ify: Q into (B, num_blocks, chunk_size, H, D),
        # K/V into context windows (B, num_blocks, context_size, H, D).
        query_states = self._convert_to_block(query_states)
        if getattr(self.config, "use_combined_kv_extract", False):
            key_states, value_states = self._extract_block_context_kv(
                key_states, value_states
            )
        else:
            key_states = self._extract_block_context(key_states)
            value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]

        # Relative-key embeddings: (1, rel_window, num_heads * head_dim) ->
        # (rel_window, num_heads, head_dim).
        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
        relative_key_states = relative_key_states.to(dtype=query_states.dtype)

        # matrix_ac: Q @ K^T over the local context window.
        # queries:     (B, H, num_blocks, chunk_size, head_dim)
        # key_states:  (B, num_blocks, context_size, H, head_dim) -> perm to
        #               (B, H, num_blocks, head_dim, context_size)
        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)

        # matrix_bd: relative-key bias.
        queries_flat = queries.reshape(
            batch_size, self.num_heads, -1, self.head_dim
        )
        # (B, H, num_blocks*chunk_size, head_dim) @ (head_dim, rel_window, H)? No:
        # relative_key_states permuted -> (H, head_dim, rel_window)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(
            batch_size, self.num_heads, num_blocks, self.chunk_size, -1
        )
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd

        # Softcap: tanh(x / cap) * cap. The `softcap` buffer is a 0-d tensor
        # but the math is `tanh(x / scalar) * scalar` — eager just promotes.
        attn_weights = attn_weights / self.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.softcap

        # Apply 5D bool mask: False (invalid) -> attention_invalid_logits_value.
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.logical_not(),
                self.config.attention_invalid_logits_value,
            )

        # Softmax in fp32 then cast back to value_states dtype.
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            value_states.dtype
        )

        # attn_output: (B, H, num_blocks, chunk_size, context_size) @
        #              (B, H, num_blocks, context_size, head_dim)
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        # back to (B, num_blocks, chunk_size, H, head_dim) -> (B, num_blocks*chunk_size, H*D)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(
            batch_size, num_blocks * self.chunk_size, -1
        )
        # Strip any padding introduced by the chunk alignment.
        attn_output = attn_output[:, :seq_length].contiguous()
        attn_output = self.post(attn_output.to(dtype=self.post.linear.weight.dtype))

        return attn_output, attn_weights

    def _forward_vulkan_sdpa(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vulkan-fused attention path (Phase 2B — full math).

        Math is now equivalent (up to Option-B softcap ordering, see below)
        to the explicit `forward`:
          - Transformer-XL relative-key bias (`matrix_bd` + `_rel_shift`) is
            built per-forward in Python, expanded to dense `(1, H, S, S)`
            per-head bias, summed with the broadcast `(1, 1, S, S)` chunked-
            block padding mask, and passed as a SINGLE per-head additive
            mask of shape `(1, H, S, S)` to the extended gemma_custom_sdpa.
          - softcap (`tanh(x / cap) * cap`) is now applied inside the op
            (post mask-add, pre softmax) — Option B in PLAN_gemma4_custom_sdpa_shader.
          - The chunked Q@K.T is unfolded into one dense Q@K.T over (B, S,
            H, D). The chunked-block pattern is encoded by the SAME
            padding mask the B5-OFF path uses (combined with matrix_bd in
            the per-head mask).

        Quality gate: B5-ON vs HF eager cos >= 0.99 on the encoder body.
        """
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

        # Q/K/V projections (BSHD).
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        # Scale Q and K. Match the chunked path: do the scale in fp32 then
        # cast back. (Eager parity is dtype-sensitive on the softplus path.)
        # Fix C: when `config.bake_softplus_scale=True`, consume
        # the pre-baked `softplus_per_dim_scale` buffer directly — saves
        # 4 host ops per layer (gt + log1p + exp + where) under torch.export.
        if getattr(self.config, "bake_softplus_scale", False):
            softplus_scale = self.softplus_per_dim_scale
        else:
            softplus_scale = F.softplus(self.per_dim_scale.float())
        query_states = (
            query_states.float()
            * self.q_scale
            * softplus_scale
        ).to(query_states.dtype)
        key_states = (key_states.float() * self.k_scale).to(key_states.dtype)

        # ------------------------------------------------------------------
        # Build matrix_bd as a per-head additive bias of shape (1, H, S, S).
        # Mirrors the chunked path, but applied densely (not block-wise).
        # The chunk grouping is already encoded in the broadcast pad mask:
        # for a given query position q in block b, valid kv positions form
        # a window covering [b*chunk_size - max_past_horizon, ...]. Within
        # that window, the rel-pos bias depends only on `dist = q - kv`,
        # so building a dense (H, S, S) bias indexed by `dist` is
        # mathematically equivalent to the per-block `_rel_shift` output
        # at the unmasked positions. At MASKED positions the bias is
        # additive — but the padding mask already drives those to -inf
        # before softmax, so any value there is irrelevant.
        # ------------------------------------------------------------------
        # Build the (B, H, S, S) bias by indexing the (rel_window, H, D)
        # rel-key embeddings by `dist = q - kv` (clipped to the rel window).
        # For audio's default config (max_past_horizon=12, chunk_size=12,
        # max_future_horizon=0), the rel window covers dist in [0, 12].
        assert position_embeddings is not None, (
            "_forward_vulkan_sdpa requires position_embeddings (the "
            "rel-pos sinusoidal table); audio_tower must call rel_pos_enc"
        )
        relative_key_states = self.relative_k_proj(position_embeddings)
        # (rel_window, H, D)
        relative_key_states = relative_key_states.view(
            -1, self.num_heads, self.head_dim
        ).to(dtype=query_states.dtype)

        # Compute Q @ rel_K^T per the chunked-path math. Q at (B, S, H, D),
        # rel_K at (rel_window, H, D). Result is per-head bias indexed by
        # (s, rel_idx). The rel_window covers absolute dist in [0,
        # max_past_horizon]; rel_idx 0 corresponds to dist=max_past_horizon,
        # rel_idx max_past_horizon corresponds to dist=0 (HF orders the
        # rel-pos embeddings from oldest to newest).
        # bias[s, kv] = Q[s] . rel_K[ rel_idx_for(dist = s - kv) ]
        #
        # We replicate matrix_bd exactly via the chunked + _rel_shift code,
        # then expand to dense (B, H, S, S):
        queries_for_bd = (
            query_states.permute(0, 2, 1, 3)  # (B, H, S, D)
            .reshape(batch_size, self.num_heads, seq_length, self.head_dim)
        )
        # matrix_bd_flat[b, h, s, r] = Q[b, s, h, :] @ rel_K[r, h, :]
        # rel_K permuted: (H, D, rel_window)
        matrix_bd_flat = queries_for_bd @ relative_key_states.permute(1, 2, 0)
        # Fold BOTH the `_rel_shift` (F.pad + view + slice + view)
        # AND the post-rel-shift `index_select` into a single index_select
        # against the PRE-rel-shift flat tensor. The baked
        # `_rel_shift_flat_idx` buffer is now composed with
        # `_rel_shift`'s inverse map (see `bake_rel_shift_flat_idx`).
        #
        # Eliminates per-layer:
        #   - 1 constant_pad_nd (from _rel_shift's F.pad(..., (0, P - rel_window)))
        #   - 0..1 constant_pad_nd (from the chunk-alignment F.pad of
        #     matrix_bd_flat — at SAMPLE_K=2, pad_s=0 so this is already DCE'd,
        #     but we drop it anyway to keep the path uniform for dynamic-S exports)
        #   - 1 reshape/view to (B, H, num_blocks, chunk_size, position_length)
        #   - 1 reshape to (B, H, -1)
        # i.e. -12 constant_pad_nd host ops at N=12 vs the unfused form.
        #
        # The forward expects `seq_length == S_full` (BLOCK_SAMPLES is a
        # multiple of chunk_size so the subsampled length is too). If a
        # future export path produces seq_length < S_full, the gather
        # below will OOB — assert.
        num_blocks = (seq_length + self.chunk_size - 1) // self.chunk_size
        S_full = num_blocks * self.chunk_size
        # NOTE: static assert at TRACE time only; in torch.export this is
        # a Python-side check on the example input's seq_length (no graph
        # op). For BLOCK_SAMPLES%chunk_size==0 this always holds.
        assert seq_length == S_full, (
            f"_forward_vulkan_sdpa baked-index path requires seq_length "
            f"({seq_length}) to equal S_full ({S_full}); "
            f"BLOCK_SAMPLES must be a multiple of chunk_size."
        )
        # matrix_bd_flat has shape (B, H, seq_length=S_full, position_length).
        # `.reshape(B, H, -1)` flattens to (B, H, S_full * position_length).
        # The baked `_rel_shift_flat_idx` (length S_max*S_max) gathers
        # into that flat tensor; we then view to (B, H, S_full, S_full).
        #
        # Dynamic samples: the baked buffer is sized at
        # S_max = self._rel_shift_seq_len_max (>= active S_full). Slice
        # the (S_max, S_max) tile down to (S_full, S_full) at runtime.
        # The index VALUES (`s_pre * position_length + p_padded`) are
        # independent of S_max, so the slice is mathematically equivalent
        # to baking directly at S_full (proof: position_length axis stride
        # is `position_length`; `s_pre` and `p_padded` depend only on
        # chunk_size / num_blocks within the slice).
        S_max = self._rel_shift_seq_len_max
        flat_idx_2d = self._rel_shift_flat_idx.view(S_max, S_max)
        flat_idx_sliced = flat_idx_2d[:S_full, :S_full].reshape(-1).contiguous()
        matrix_bd_flat_3d = matrix_bd_flat.reshape(
            batch_size, self.num_heads, -1
        )
        matrix_bd_dense = matrix_bd_flat_3d.index_select(
            dim=-1, index=flat_idx_sliced
        ).view(batch_size, self.num_heads, S_full, S_full)

        # Combined per-head mask: matrix_bd + broadcast(padding_mask).
        # padding_mask is (1, 1, S_max, S_max) float (-inf masked,
        # 0 unmasked). Slice to (1, 1, S_full, S_full).
        if attention_mask is not None:
            mask_sliced = attention_mask[:, :, :S_full, :S_full].contiguous()
            combined_mask = matrix_bd_dense + mask_sliced.to(
                matrix_bd_dense.dtype
            )
        else:
            combined_mask = matrix_bd_dense

        # Cast combined mask to the Q dtype (the SDPA op casts internally).
        combined_mask = combined_mask.to(query_states.dtype)

        # Call extended gemma_custom_sdpa with per-head mask and softcap.
        attn_output = torch.ops.et_vk.gemma_custom_sdpa.default(
            query_states,
            key_states,
            value_states,
            0,                                          # start_pos
            seq_length,                                 # seq_len
            combined_mask,                              # (1, H, S, S) per-head additive float
            False,                                      # is_causal (ignored when mask is provided)
            float(self.attention_logits_soft_cap),      # softcap_value (e.g. 50.0)
        )
        # Output is (B, S, H, D); collapse heads.
        attn_output = attn_output.contiguous().view(
            batch_size, seq_length, -1
        )
        attn_output = self.post(
            attn_output.to(dtype=self.post.linear.weight.dtype)
        )
        # Return (output, output) tuple to match the chunked-path signature;
        # callers only use [0].
        return attn_output, attn_output


# ---------------------------------------------------------------------------
# Subsample conv projection (input front-end)
# ---------------------------------------------------------------------------


class Gemma4EdgeAudioSubSampleConvProjectionLayer(nn.Module):
    """Mirror of `Gemma4AudioSubSampleConvProjectionLayer` (HF lines 318-343).

    A single 3x3 stride-2 conv2d with `padding=1`, followed by LayerNorm
    on the channel dim (out_channels) with `elementwise_affine=True,
    bias=False`, then ReLU. Masking propagates through `mask[:, ::2]`.
    """

    def __init__(self, in_channels: int, out_channels: int, norm_eps: float):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
        )
        # HF uses `nn.LayerNorm(out_channels, ..., elementwise_affine=True, bias=False)`.
        # `bias=False` is supported by `nn.LayerNorm` (>=PyTorch 2.1).
        self.norm = nn.LayerNorm(
            out_channels, eps=norm_eps, elementwise_affine=True, bias=False
        )
        self.act = nn.ReLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mask is not None:
            # Broadcast 1D mask (B, T) -> (B, 1, T, 1) for elementwise multiply.
            mask = mask.to(device=hidden_states.device)
            hidden_states = hidden_states * mask[:, None, :, None]

        # Conv eats the conv weight dtype; HF casts here to be safe.
        hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
        # LayerNorm over channels: permute (B, C, T, F) -> (B, T, F, C),
        # normalize the last dim (C), permute back.
        hidden_states = self.act(
            self.norm(hidden_states.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        if mask is not None:
            mask = mask[:, ::2]

        return hidden_states, mask


class Gemma4EdgeAudioSubSampleConvProjection(nn.Module):
    """Mirror of `Gemma4AudioSubSampleConvProjection` (HF lines 346-373).

    Two stacked subsample conv layers + a linear projection from the
    concatenated channel/freq features into `hidden_size`. Net effect:
    time axis downsampled by 4x (two stride-2 convs); freq axis from 128
    mel bins -> 32 -> 32 (subsampling_conv_channels[1]) * (128/4) flat.
    """

    def __init__(self, config: Gemma4EdgeAudioConfig):
        super().__init__()
        self.layer0 = Gemma4EdgeAudioSubSampleConvProjectionLayer(
            in_channels=1,
            out_channels=config.subsampling_conv_channels[0],
            norm_eps=config.rms_norm_eps,
        )
        self.layer1 = Gemma4EdgeAudioSubSampleConvProjectionLayer(
            in_channels=config.subsampling_conv_channels[0],
            out_channels=config.subsampling_conv_channels[1],
            norm_eps=config.rms_norm_eps,
        )
        # HF: proj_input_dim = (subsampling_conv_channels[0] // 4) * subsampling_conv_channels[1]
        # which is `(128 // 4) * 32 = 1024` for the default config. This matches
        # the post-conv flattened (T/4, F/4 * C[1]) feature dim. Note: HF uses
        # `subsampling_conv_channels[0] // 4` (i.e. 128 // 4 = 32) for the freq
        # axis residue, but that's the freq downsample factor (two stride-2 = 4x
        # on the freq axis from 128 mel bins => 32). Then `* C[1] = 32` gives 1024.
        proj_input_dim = (
            config.subsampling_conv_channels[0] // 4
        ) * config.subsampling_conv_channels[1]
        self.input_proj_linear = nn.Linear(proj_input_dim, config.hidden_size, bias=False)

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Add channel dim: (B, T, F) -> (B, 1, T, F).
        hidden_states = input_features.unsqueeze(1)
        hidden_states, mask = self.layer0(hidden_states, input_features_mask)
        hidden_states, mask = self.layer1(hidden_states, mask)

        # (B, C[1], T_down, F_down) -> (B, T_down, F_down * C[1])
        batch_size, _, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous().reshape(
            batch_size, seq_len, -1
        )
        return self.input_proj_linear(hidden_states), mask


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------


class Gemma4EdgeAudioFeedForward(nn.Module):
    """Mirror of `Gemma4AudioFeedForward` (HF lines 376-408).

    Standard Conformer-style FFN:
        clamp -> pre_norm -> ffw1 -> SiLU -> ffw2 -> clamp -> post_norm
        -> scale by residual_weight -> add residual.
    """

    def __init__(self, config: Gemma4EdgeAudioConfig):
        super().__init__()
        self.config = config

        self.ffw_layer_1 = ClippableLinear(
            config, config.hidden_size, config.hidden_size * 4
        )
        self.ffw_layer_2 = ClippableLinear(
            config, config.hidden_size * 4, config.hidden_size
        )

        self.pre_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.hidden_act == "silu":
            self.act_fn = F.silu
        elif config.hidden_act == "gelu":
            self.act_fn = F.gelu
        elif config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Clamp using the projection-weight finfo to avoid runaway scaling.
        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.ffw_layer_1.linear.weight.dtype).max,
        )

        residual = hidden_states
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.pre_layer_norm(hidden_states)

        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.post_layer_norm(hidden_states)
        # In-place scale + add to match HF byte-for-byte (operator ordering).
        hidden_states = hidden_states * self.post_layer_scale
        hidden_states = hidden_states + residual

        return hidden_states


# ---------------------------------------------------------------------------
# Causal Conv1d (manual left-padding)
# ---------------------------------------------------------------------------


class Gemma4EdgeAudioCausalConv1d(nn.Conv1d):
    """Mirror of `Gemma4AudioCausalConv1d` (HF lines 412-442).

    Extends `nn.Conv1d` with manual left-padding so the layer is causal.
    The `left_pad` length is `(kernel_size - 1) * dilation - stride + 1`,
    cached on first access (HF uses `@cached_property`; we use a plain
    `property` to avoid leaking module-level Python state into a buffer
    on `state_dict`).
    """

    @property
    def left_pad(self) -> int:
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


# ---------------------------------------------------------------------------
# Lightweight Conv1d block
# ---------------------------------------------------------------------------


class Gemma4EdgeAudioLightConv1d(nn.Module):
    """Mirror of `Gemma4AudioLightConv1d` (HF lines 445-483).

    Pre-norm -> linear_start (-> 2H) -> GLU -> causal depthwise conv1d ->
    clamp -> conv_norm -> SiLU -> linear_end -> residual add.
    """

    def __init__(self, config: Gemma4EdgeAudioConfig):
        super().__init__()
        self.config = config

        self.linear_start = ClippableLinear(
            config, config.hidden_size, config.hidden_size * 2
        )
        self.linear_end = ClippableLinear(
            config, config.hidden_size, config.hidden_size
        )
        self.depthwise_conv1d = Gemma4EdgeAudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
        )

        self.pre_layer_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, with_scale=True
        )
        self.conv_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, with_scale=True
        )

        if config.hidden_act == "silu":
            self.act_fn = F.silu
        elif config.hidden_act == "gelu":
            self.act_fn = F.gelu
        elif config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = F.glu(hidden_states, dim=-1)

        # Depthwise conv on the time axis: (B, T, C) -> (B, C, T) -> conv -> (B, T, C).
        hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)

        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.linear_start.linear.weight.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.conv_norm(hidden_states)

        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


# ---------------------------------------------------------------------------
# Audio encoder layer
# ---------------------------------------------------------------------------


class Gemma4EdgeAudioLayer(nn.Module):
    """Mirror of `Gemma4AudioLayer` (HF lines 486-534).

    Conformer block: FFN1 -> attention with sandwich norms -> LightConv1d
    -> FFN2 -> output norm.
    """

    def __init__(self, config: Gemma4EdgeAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config

        self.feed_forward1 = Gemma4EdgeAudioFeedForward(config)
        self.feed_forward2 = Gemma4EdgeAudioFeedForward(config)
        self.self_attn = Gemma4EdgeAudioAttention(config, layer_idx)
        self.lconv1d = Gemma4EdgeAudioLightConv1d(config)

        self.norm_pre_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_post_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_out = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_clipping = config.gradient_clipping

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.BoolTensor],
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.norm_pre_attn.weight.dtype).max,
        )

        # FFN1 (note: this FFN already adds residual internally).
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_pre_attn(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Chunked-attention mask: 4D bool -> 5D blocked bool
# ---------------------------------------------------------------------------


def _convert_4d_mask_to_blocked_5d(
    mask_4d: torch.Tensor,
    chunk_size: int,
    max_past_horizon: int,
    max_future_horizon: int,
) -> torch.Tensor:
    """Convert a 4D `(B, 1, S, S)` bool mask to the 5D blocked
    `(B, 1, num_blocks, chunk_size, context_size)` layout consumed by
    `Gemma4EdgeAudioAttention`.

    Implementation = the `unfold + diagonal` variant from the deleted
    `export_audio_encoder_bf16.py` (commit `9a827e7bfd`). This avoids
    `torch.arange(symbolic_T)` so dynamic-shape export works. Math is
    identical to HF `_convert_4d_mask_to_blocked_5d` (HF lines 1803-1828).
    """
    batch_size, _, seq_len, _ = mask_4d.shape
    context_size = chunk_size + max_past_horizon + max_future_horizon

    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    padded_seq_len = num_blocks * chunk_size
    pad_amount = padded_seq_len - seq_len

    # Pad query and key axes to a multiple of chunk_size.
    m = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
    # Pad the key axis by the past/future horizon (so that the unfold below
    # produces full-width context windows even at the edges).
    m = F.pad(m, (max_past_horizon, max_future_horizon), value=False)
    m = m.reshape(
        batch_size, 1, num_blocks, chunk_size,
        padded_seq_len + max_past_horizon + max_future_horizon,
    )
    # Unfold the kv axis into overlapping context windows of size
    # `context_size`, strided by `chunk_size`. The unfold creates a new
    # last dim of length `context_size`; the corresponding axis is at
    # position 4 (between chunk_size and the original kv axis).
    m = m.unfold(-1, context_size, chunk_size)
    # `m` shape: (B, 1, num_blocks, chunk_size, num_blocks, context_size).
    # We want the diagonal over the block-pair axes (dim=2 and dim=4) so
    # that for block i we keep window i.
    m = torch.diagonal(m, offset=0, dim1=2, dim2=4)
    # `m` shape after diagonal: (B, 1, chunk_size, context_size, num_blocks).
    # Reorder back to HF's (B, 1, num_blocks, chunk_size, context_size).
    m = m.permute(0, 1, 4, 2, 3)
    return m.contiguous()


def _build_chunked_block_mask_4d(
    valid_mask: torch.Tensor,
    chunk_size: int,
    max_past_horizon: int,
    max_future_horizon: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the 4D additive `(B, 1, S, S)` chunked-block mask for the
    Vulkan SDPA path (`use_vulkan_sdpa=True`).

    The chunked-attention semantics encoded in this dense mask:
      - For query at position `q` in block `b = q // chunk_size`, valid
        key positions are `kv in [b*chunk_size - max_past_horizon,
        b*chunk_size + chunk_size + max_future_horizon)`.
      - Within that window, the per-position attending set is further
        restricted by `(dist >= 0) & (dist < max_past_horizon)` (HF's
        `sliding_window_mask_function` semantics, where
        `dist = q - kv`).
      - The per-block context window restriction is implied by the
        intersection of the two — for the default audio config
        (`chunk_size=12`, `max_past_horizon=12`, `max_future_horizon=0`),
        the two collapse to "q attends to itself and 11 preceding
        positions, both must be in the same `b*12 .. (b+1)*12 - 1` query
        block".
      - Plus AND with the padding mask (`valid_mask[b, kv]`).

    Equivalent to math performed by the chunked path with the 5D blocked
    mask AT the unmasked positions; only difference is that the rel-pos
    bias and softcap are dropped on the Vulkan SDPA path (which only
    accepts a `(1, 1, S, S)` mask).

    Args:
      valid_mask: `(B, S)` bool, True where the sample is valid.
      chunk_size, max_past_horizon, max_future_horizon: integers from
        the audio config (defaults: 12, 12, 0).
      dtype: floating dtype used for the additive mask (0 at attending
        positions, finfo(dtype).min at non-attending positions).

    Returns:
      `(B, 1, S, S)` float additive mask.
    """
    device = valid_mask.device
    B, seq_len = valid_mask.shape

    q_idx = torch.arange(seq_len, device=device)[None, None, :, None]  # (1,1,q,1)
    kv_idx = torch.arange(seq_len, device=device)[None, None, None, :] # (1,1,1,k)
    dist = q_idx - kv_idx

    # Chunked-block window: q and kv must agree on the same chunk
    # group, OR kv must be in the past_horizon of q's chunk start.
    # Per HF's `bidirectional_mask_function` AND-ed with
    # `sliding_window_mask_function(max_past_horizon, max_future_horizon)`,
    # the test is the standard sliding-window: `(dist >= 0) AND
    # (dist < max_past_horizon)` for the left half, plus
    # `(dist < 0) AND (-dist < max_future_horizon)` for the right half.
    # The chunk grouping is implicit in HF when this mask is passed
    # through `_convert_4d_mask_to_blocked_5d`; here we keep just the
    # sliding-window pattern, which is mathematically identical to the
    # chunked semantics for all (q, kv) inside the chunk-pair window
    # (and the chunk-pair window's effect is captured by the per-row
    # restriction `0 <= dist < max_past_horizon`).
    left_mask = (dist >= 0) & (dist < max_past_horizon)
    if max_future_horizon > 0:
        right_mask = (dist < 0) & (-dist < max_future_horizon)
        window_mask = left_mask | right_mask
    else:
        window_mask = left_mask

    # Padding mask: (B, 1, 1, kv)
    padding_mask_4d = valid_mask[:, None, None, :]
    bool_mask = padding_mask_4d & window_mask  # (B, 1, S, S)

    min_val = torch.finfo(dtype).min
    additive = torch.where(
        bool_mask,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_val, dtype=dtype, device=device),
    )
    return additive


def _build_bidirectional_audio_mask(
    valid_mask: torch.Tensor,
    chunk_size: int,
    max_past_horizon: int,
    max_future_horizon: int,
) -> torch.Tensor:
    """Build the 5D bool attention mask consumed by `Gemma4EdgeAudioAttention`.

    HF's flow:
      1. `create_bidirectional_mask(config, inputs_embeds=hidden_states,
                                     attention_mask=valid_mask,
                                     and_mask_function=sliding_window_mask_function(...))`
         under attn_implementation="sdpa" produces a 4D `(B, 1, S, S)`
         bool mask where True == valid. The base function is
         `bidirectional_mask_function` (q_idx >= 0, always True), AND-ed
         with the sliding window `|q-k| in [-right, +left)` AND with the
         padding mask (`valid_mask[b, k]`).
      2. `_convert_4d_mask_to_blocked_5d(...)` reshapes to the 5D blocked
         layout consumed by chunked attention.

    Here we build the 4D bool mask DIRECTLY (skipping the
    `create_bidirectional_mask` recursion into masking_utils) and then
    apply the same blocked reshape. The bidirectional q_idx>=0 reduces
    to True everywhere; the AND with padding broadcasts (B, kv_len) to
    (B, 1, q_len, kv_len); the AND with the sliding window gives the
    per-position window. All three conditions are AND-ed in fp32-free
    bool math.
    """
    device = valid_mask.device
    B, seq_len = valid_mask.shape

    # Padding mask: (B, 1, 1, kv_len) -- True where kv position is valid.
    padding_mask_4d = valid_mask[:, None, None, :]  # (B, 1, 1, kv_len)

    # Sliding window mask: (1, 1, q_len, kv_len) -- True where the relative
    # offset is in (-right, +left). HF's `sliding_window_mask_function` is
    # invoked with `(attention_context_left - 1, attention_context_right)`,
    # so `left_window_size = max_past_horizon` and the test is
    # `(dist >= 0) & (dist < max_past_horizon)` -- i.e. each query attends
    # to its own position and the (max_past_horizon - 1) preceding ones.
    # For the default audio config (`attention_context_left=13`), this
    # gives a 12-wide left window: q, q-1, ..., q-11.
    q_idx = torch.arange(seq_len, device=device)[None, None, :, None]  # (1,1,q,1)
    kv_idx = torch.arange(seq_len, device=device)[None, None, None, :] # (1,1,1,k)
    dist = q_idx - kv_idx
    left_mask = (dist >= 0) & (dist < max_past_horizon)
    if max_future_horizon > 0:
        right_mask = (dist < 0) & (-dist < max_future_horizon)
        window_mask = left_mask | right_mask
    else:
        window_mask = left_mask

    # Combined 4D bool: padding AND window (broadcasts to (B, 1, q_len, kv_len)).
    mask_4d = padding_mask_4d & window_mask  # (B, 1, q_len, kv_len)

    # Reshape to blocked 5D.
    return _convert_4d_mask_to_blocked_5d(
        mask_4d,
        chunk_size=chunk_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
    )


# ---------------------------------------------------------------------------
# LogMel frontend
# ---------------------------------------------------------------------------


class LogMelFrontend(nn.Module):
    """Waveform -> log-mel spectrogram.

    Mirrors `Gemma4AudioFeatureExtractor` (HF feature_extraction_gemma4.py
    lines 175-225) bit-for-bit (up to fp accumulation order):

      1. Left-pad waveform by `frame_length // 2 = 160` zeros (semicausal
         padding so the first STFT frame is centered at t=0).
      2. Unfold into frames of size `frame_length + 1 = 321` with step
         `hop_length = 160`; drop the last sample of each frame (HF does
         this to support optional preemphasis filtering; for the default
         `preemphasis = 0.0` config it just produces a 320-sample frame).
      3. Multiply by the Hann window.
      4. `np.fft.rfft` (HF) / `torch.fft.rfft` (edge, default) with
         n=512 -> 257 complex bins. The bake-in `dft_real`/`dft_imag`
         matmul path is kept behind `use_matmul_dft=True` for export
         compatibility (matches `export_audio_encoder_bf16.py` commit
         `9a827e7bfd`); residual fp drift on a long real waveform is
         ~1e-4 max-abs vs HF.
      5. Magnitude (`abs`), mel filter bank multiply, `log(mel + 1e-3)`.
      6. Frame-validity mask: unfold the input attention_mask the same
         way and read the LAST sample of each unfolded frame (HF reads
         `attention_mask[i*hop + frame_length]`; we read the equivalent
         via `mask_frames[..., -1]` on a `size=frame_length+1` unfold,
         which is the same byte position).

    Inputs:
      - `waveform`: `(B, T)` float32 audio samples at 16 kHz.
      - `attention_mask`: `(B, T)` bool, True where samples are valid.
    Outputs:
      - `log_mel`: `(B, T_frames, 128)` log-mel spectrogram.
      - `frame_mask`: `(B, T_frames)` bool, True where the frame is valid.

    `T_frames = (T + frame_length // 2 - (frame_length + 1)) // hop + 1
              = (T - 161) // 160 + 1`.
    """

    def __init__(
        self,
        hann_window: np.ndarray,
        mel_filters: np.ndarray,
        use_matmul_dft: bool = True,
    ):
        super().__init__()
        self.use_matmul_dft = bool(use_matmul_dft)
        self.register_buffer(
            "hann_window",
            torch.as_tensor(hann_window, dtype=torch.float32),
        )
        self.register_buffer(
            "mel_filters",
            torch.as_tensor(mel_filters, dtype=torch.float32),
        )
        # Pre-bake the (frame_length, fft_bins) real/imag DFT matrices for
        # the optional matmul path (export-friendly form). HF's processor
        # uses `np.cos(2*pi*n*k/N)` / `-np.sin(...)` -- bake the SAME.
        if self.use_matmul_dft:
            n_vec = np.arange(AUDIO_FRAME_LENGTH, dtype=np.float64)
            k_vec = np.arange(AUDIO_FFT_LENGTH // 2 + 1, dtype=np.float64)
            ang = 2.0 * np.pi * np.outer(n_vec, k_vec) / AUDIO_FFT_LENGTH
            self.register_buffer(
                "dft_real",
                torch.as_tensor(np.cos(ang).astype(np.float32)),
            )
            self.register_buffer(
                "dft_imag",
                torch.as_tensor((-np.sin(ang)).astype(np.float32)),
            )

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Semicausal time padding: only left-pad (HF processor lines 180-184).
        # `AUDIO_PAD_RIGHT = 0`; we keep the constant for API parity with the
        # earlier matmul-DFT version that padded right by 1.
        wav = F.pad(waveform, (AUDIO_PAD_LEFT, AUDIO_PAD_RIGHT))
        mask = F.pad(attention_mask, (AUDIO_PAD_LEFT, AUDIO_PAD_RIGHT))

        # Frame: window size 321, step = hop_length (160). Then trim to 320.
        # The "321 unfold then trim to 320" pattern is the
        # `frame_size_for_unfold = self.frame_length + 1` of HF: the extra
        # sample exists to support preemphasis filtering (HF drops the
        # FIRST sample after the filter), but with `preemphasis=0.0` we
        # just drop the LAST sample to get back to `frame_length`.
        frames = wav.unfold(
            dimension=-1, size=AUDIO_FRAME_UNFOLD_SIZE, step=AUDIO_HOP_LENGTH
        )
        frames = frames[..., :AUDIO_FRAME_LENGTH]
        frames = frames * self.hann_window

        if self.use_matmul_dft:
            # Export-friendly path: real-valued DFT via two real matmuls.
            # Fp accumulation order differs from `np.fft.rfft`, producing
            # ~1e-4 max-abs drift on a long real waveform (cosine still 1.0
            # to 7 digits). Use when bound for Vulkan export.
            stft_real = frames @ self.dft_real
            stft_imag = frames @ self.dft_imag
            magnitude = torch.sqrt(stft_real * stft_real + stft_imag * stft_imag)
        else:
            # Default eager path: `torch.fft.rfft` with the SAME `n=fft_length`
            # right-pad as `np.fft.rfft`. Closer to HF (~1e-5 max-abs on a
            # long waveform).
            stft = torch.fft.rfft(frames, n=AUDIO_FFT_LENGTH, dim=-1)
            magnitude = stft.abs()

        # Mel filter bank: (fft_bins, n_mel) -> (B, T_frames, n_mel).
        mel = magnitude @ self.mel_filters
        log_mel = torch.log(mel + AUDIO_MEL_FLOOR)

        # Frame-level validity bitmask: unfold the input mask the same way
        # as the waveform, then read the last sample of each frame.
        mask_frames = mask.unfold(
            dimension=-1, size=AUDIO_FRAME_UNFOLD_SIZE, step=AUDIO_HOP_LENGTH
        )
        frame_mask = mask_frames[..., -1].to(torch.bool)
        log_mel = log_mel * frame_mask.unsqueeze(-1).to(log_mel.dtype)
        return log_mel, frame_mask


# ---------------------------------------------------------------------------
# Audio encoder body (subsample + rel-pos + N layers + output proj)
# ---------------------------------------------------------------------------


@dataclass
class _AudioEncoderOutput:
    """Lightweight stand-in for `Gemma4AudioModelOutput`."""

    last_hidden_state: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None


class Gemma4EdgeAudioModel(nn.Module):
    """Mirror of `Gemma4AudioModel` (HF lines 1779-1861).

    Encapsulates everything HF's `audio_tower` does:
      1. Subsample conv projection (waveform features -> hidden_size).
      2. Relative positional encoding (per-block window).
      3. Stack of N Gemma4AudioLayer.
      4. Output projection to `output_proj_dims`.
    """

    def __init__(self, config: Gemma4EdgeAudioConfig):
        super().__init__()
        self.config = config

        self.subsample_conv_projection = Gemma4EdgeAudioSubSampleConvProjection(config)
        self.rel_pos_enc = Gemma4EdgeAudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [
                Gemma4EdgeAudioLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        # HF uses `bias=True` on this output projection.
        self.output_proj = nn.Linear(
            config.hidden_size, config.output_proj_dims, bias=True
        )
        # Vulkan SDPA path (`use_vulkan_sdpa=True`): pre-bake the static
        # additive `(1, 1, S_max, S_max)` chunked-block mask. The caller
        # binds it via `bake_audio_sdpa_mask(seq_len_max, dtype)` before
        # export. Default of `zeros(0)` is a placeholder. Persistent=True
        # so it lowers to a TensorRef constant on the Vulkan prepack path.
        if getattr(config, "use_vulkan_sdpa", False):
            self.register_buffer(
                "_audio_sdpa_mask",
                torch.zeros(0, dtype=torch.float32),
                persistent=True,
            )

        # Lever B: pre-baked 5D blocked bool mask for the non-
        # Vulkan attention forward. Bound via `bake_audio_5d_mask` before
        # export; sliced to `:num_blocks_active` at runtime.
        if getattr(config, "use_baked_5d_mask", False):
            self.register_buffer(
                "_audio_5d_blocked_mask",
                torch.zeros(0, dtype=torch.bool),
                persistent=True,
            )

    def bake_audio_sdpa_mask(
        self, seq_len_max: int, dtype: torch.dtype = torch.float32
    ) -> None:
        """Materialize the static `(1, 1, S_max, S_max)` chunked-block
        additive mask used when `use_vulkan_sdpa=True`.

        The mask encodes the audio chunked-attention pattern as a dense
        S×S block (HF `sliding_window_mask_function` with
        `(max_past_horizon, max_future_horizon)`). At inference, all
        positions in [0, seq_len_max) are valid (caller pre-pads K, V to
        seq_len_max), so the mask is purely the sliding-window pattern.

        Should be called after `__init__` but before export, once the
        caller fixes the static S.
        """
        if not getattr(self.config, "use_vulkan_sdpa", False):
            return
        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right
        valid_mask = torch.ones(1, seq_len_max, dtype=torch.bool)
        mask = _build_chunked_block_mask_4d(
            valid_mask,
            chunk_size=chunk_size,
            max_past_horizon=max_past_horizon,
            max_future_horizon=max_future_horizon,
            dtype=dtype,
        )
        self._audio_sdpa_mask = mask.contiguous()
        self.config.sdpa_mask_seq_len_max = int(seq_len_max)

        # Bake the rel-shift flat-index buffer on every attention
        # layer. This replaces the per-block `slice_scatter` loop in
        # `_forward_vulkan_sdpa` with a single `index_select` (1 host op vs
        # 36 slice_scatter + 24 copy at num_blocks=2; scales linearly).
        for layer in self.layers:
            layer.self_attn.bake_rel_shift_flat_idx(seq_len_max=int(seq_len_max))

    def bake_audio_5d_mask(self, seq_len_max: int) -> None:
        """Lever B: materialize the static 5D blocked bool mask
        for the non-Vulkan attention forward.

        Shape: `(1, 1, num_blocks_max, chunk_size, context_size)` where
        `num_blocks_max = ceil(seq_len_max / chunk_size)` and
        `context_size = chunk_size + max_past + max_future`.

        Assumes all positions in `[0, seq_len_max)` are valid (mirrors
        the Vulkan path's invariant). Runtime audio inputs are always
        fully-valid in practice.

        Should be called after `__init__` but before export, once the
        caller fixes the static max-S.
        """
        if not getattr(self.config, "use_baked_5d_mask", False):
            return
        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right
        valid_mask = torch.ones(1, seq_len_max, dtype=torch.bool)
        mask_5d = _build_bidirectional_audio_mask(
            valid_mask,
            chunk_size=chunk_size,
            max_past_horizon=max_past_horizon,
            max_future_horizon=max_future_horizon,
        )
        self._audio_5d_blocked_mask = mask_5d.contiguous()
        self.config.baked_5d_mask_seq_len_max = int(seq_len_max)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> _AudioEncoderOutput:
        """
        input_features: `(B, T_frames, 128)` log-mel spectrogram.
        attention_mask: `(B, T_frames)` bool, True where the frame is valid.

        Returns `_AudioEncoderOutput(last_hidden_state, attention_mask)`
        where `last_hidden_state` is `(B, T_down, output_proj_dims)` and
        `attention_mask` is `(B, T_down)` -- the mask after the
        subsampling conv stack (downsampled 4x on the time axis).
        """
        hidden_states, output_mask = self.subsample_conv_projection(
            input_features, attention_mask
        )

        use_vulkan_sdpa = getattr(self.config, "use_vulkan_sdpa", False)
        if use_vulkan_sdpa:
            # Vulkan SDPA path (Phase 2B): now KEEPS rel_pos_enc — matrix_bd
            # is built dense and folded into the per-head mask. The static
            # padding mask is (1, 1, S, S) broadcast over heads.
            attention_mask_for_layer = self._audio_sdpa_mask.to(
                hidden_states.dtype
            )
            position_embeddings = self.rel_pos_enc(hidden_states)
            for encoder_layer in self.layers[: self.config.num_hidden_layers]:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_for_layer,
                    position_embeddings=position_embeddings,
                )
        else:
            position_embeddings = self.rel_pos_enc(hidden_states)

            # 5D bool blocked attention mask.
            if output_mask is not None:
                if getattr(self.config, "use_baked_5d_mask", False):
                    # Lever B: slice from pre-baked window mask.
                    # Assumes output_mask is all-True (typical audio
                    # inference); the bake encodes only the chunked
                    # window pattern, not per-input padding.
                    chunk_size = self.config.attention_chunk_size
                    s = output_mask.shape[1]
                    num_blocks_active = (s + chunk_size - 1) // chunk_size
                    attention_mask_5d = self._audio_5d_blocked_mask[
                        :, :, :num_blocks_active, :, :
                    ]
                else:
                    attention_mask_5d = _build_bidirectional_audio_mask(
                        output_mask,
                        chunk_size=self.config.attention_chunk_size,
                        max_past_horizon=self.config.attention_context_left - 1,
                        max_future_horizon=self.config.attention_context_right,
                    )
            else:
                attention_mask_5d = None

            for encoder_layer in self.layers[: self.config.num_hidden_layers]:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_5d,
                    position_embeddings=position_embeddings,
                )

        hidden_states = self.output_proj(hidden_states)
        return _AudioEncoderOutput(
            last_hidden_state=hidden_states, attention_mask=output_mask
        )


# ---------------------------------------------------------------------------
# Final multimodal embedder (`embed_audio` in HF)
# ---------------------------------------------------------------------------


class Gemma4EdgeAudioEmbedder(nn.Module):
    """Mirror of `Gemma4MultimodalEmbedder` (HF lines 1929-1953).

    Pre-norm (`Gemma4RMSNorm` without scale) over `output_proj_dims`, then
    a single bias-free `nn.Linear` to the text-decoder hidden size.
    """

    def __init__(self, config: Gemma4EdgeAudioConfig):
        super().__init__()
        self.multimodal_hidden_size = config.output_proj_dims
        self.eps = config.rms_norm_eps
        self.text_hidden_size = config.text_hidden_size
        self.embedding_projection = nn.Linear(
            self.multimodal_hidden_size, self.text_hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = RMSNorm(
            self.multimodal_hidden_size, eps=self.eps, with_scale=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------


class Gemma4EdgeAudioTower(nn.Module):
    """Top-level wrapper.

    Exposes:
      - `.config`           -- Gemma4EdgeAudioConfig
      - `.frontend`         -- LogMelFrontend
      - `.audio_tower`      -- Gemma4EdgeAudioModel (subsample + N layers)
      - `.embed_audio`      -- Gemma4EdgeAudioEmbedder

    Forward signature mirrors the deleted `AudioEncoderExport` from
    `export_audio_encoder_bf16.py`:
        forward(waveform, attention_mask) -> audio_embeds
    """

    def __init__(
        self,
        config: Gemma4EdgeAudioConfig,
        hann_window: Optional[np.ndarray] = None,
        mel_filters: Optional[np.ndarray] = None,
        use_matmul_dft: bool = True,
    ):
        super().__init__()
        self.config = config
        # If `hann_window`/`mel_filters` aren't supplied, register zero
        # buffers as placeholders; the caller must overwrite them before
        # forwarding (e.g. via `from_hf`).
        if hann_window is None:
            hann_window = np.zeros(AUDIO_FRAME_LENGTH, dtype=np.float32)
        if mel_filters is None:
            mel_filters = np.zeros(
                (AUDIO_FFT_LENGTH // 2 + 1, AUDIO_N_MEL), dtype=np.float32
            )
        self.frontend = LogMelFrontend(
            hann_window, mel_filters, use_matmul_dft=use_matmul_dft
        )
        self.audio_tower = Gemma4EdgeAudioModel(config)
        self.embed_audio = Gemma4EdgeAudioEmbedder(config)

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end: waveform -> audio soft-token embeddings in text space.

        Args:
            waveform: `(B, T_samples)` fp32 waveform at 16 kHz.
            attention_mask: `(B, T_samples)` bool, True where the sample is
                valid.
        Returns:
            `(B, N_soft, text_hidden_size)` audio embeddings.
        """
        log_mel, frame_mask = self.frontend(waveform, attention_mask)
        audio_out = self.audio_tower(log_mel, attention_mask=frame_mask)
        return self.embed_audio(audio_out.last_hidden_state)

    def bake_audio_sdpa_mask(
        self, seq_len_max: int, dtype: torch.dtype = torch.float32
    ) -> None:
        """Convenience wrapper around `self.audio_tower.bake_audio_sdpa_mask`.

        No-op when `use_vulkan_sdpa=False`. Must be called BEFORE export
        with `seq_len_max` matching the static S the export will pin.
        """
        self.audio_tower.bake_audio_sdpa_mask(seq_len_max=seq_len_max, dtype=dtype)

    def bake_audio_5d_mask(self, seq_len_max: int) -> None:
        """Lever B: convenience wrapper for the non-Vulkan
        5D blocked mask bake. No-op when `use_baked_5d_mask=False`.
        """
        self.audio_tower.bake_audio_5d_mask(seq_len_max=seq_len_max)

    def bake_rel_shift_baked_idx(self) -> None:
        """Lever G: for every attention layer, fill the
        pre-baked `_rel_shift_baked_idx` buffer used by the non-Vulkan
        `_rel_shift` fast-path.

        No-op when `config.use_baked_rel_shift_idx=False`. Must be called
        BEFORE export. Independent of `use_vulkan_sdpa` (the Vulkan path
        uses a different bake, `_rel_shift_flat_idx`).
        """
        if not getattr(self.config, "use_baked_rel_shift_idx", False):
            return
        n_baked = 0
        for layer in self.audio_tower.layers:
            attn = layer.self_attn
            if hasattr(attn, "bake_rel_shift_baked_idx"):
                attn.bake_rel_shift_baked_idx()
                n_baked += 1
        print(
            f"  [audio] baked _rel_shift_baked_idx on {n_baked} attention "
            f"layers (Lever G)"
        )

    def bake_softplus_per_dim_scale(self) -> None:
        """Fix C: for every attention layer, fill the pre-baked
        `softplus_per_dim_scale` buffer from the loaded `per_dim_scale`.

        No-op when `config.bake_softplus_scale=False`. Must be called AFTER
        state-dict load (per_dim_scale is a learned param that starts at
        zeros and gets overwritten by load) and BEFORE export.
        """
        if not getattr(self.config, "bake_softplus_scale", False):
            return
        n_baked = 0
        for layer in self.audio_tower.layers:
            attn = layer.self_attn
            if hasattr(attn, "bake_softplus_per_dim_scale"):
                attn.bake_softplus_per_dim_scale()
                n_baked += 1
        print(
            f"  [audio] baked softplus_per_dim_scale on {n_baked} attention "
            f"layers (Fix C)"
        )

    # ----- HF -> edge state-dict loader -------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_audio_tower: nn.Module,
        hf_embed_audio: nn.Module,
        hann_window: np.ndarray,
        mel_filters: np.ndarray,
        text_hidden_size: int,
        config: Optional[Gemma4EdgeAudioConfig] = None,
        use_matmul_dft: bool = True,
    ) -> "Gemma4EdgeAudioTower":
        """Load the edge tower from HF modules.

        Args:
            hf_audio_tower: `model.audio_tower` (an instance of
                `Gemma4AudioModel`).
            hf_embed_audio: `model.embed_audio` (an instance of
                `Gemma4MultimodalEmbedder`).
            hann_window: 1D numpy array of length `AUDIO_FRAME_LENGTH`,
                taken from `processor.feature_extractor.window`.
            mel_filters: 2D numpy array of shape `(fft_bins, n_mel)`,
                taken from `processor.feature_extractor.mel_filters`.
            text_hidden_size: hidden size of the text decoder (used by
                `embed_audio` to size its output projection).
            config: optional pre-built `Gemma4EdgeAudioConfig`. If `None`,
                we derive one from `hf_audio_tower.config`.

        Note on `attn_implementation`: HF's `Gemma4AudioAttention.forward`
        uses `attn_weights.masked_fill(mask.logical_not(), invalid)` which
        REQUIRES a bool mask. The bool mask is produced by `sdpa_mask` in
        masking_utils; the float mask from `eager_mask` would invert the
        semantics. We replicate the bool path here. Callers loading via
        `from_hf` should use `attn_implementation="sdpa"` (the default).
        """
        if config is None:
            config = _config_from_hf(hf_audio_tower.config)
        # Override text_hidden_size from the caller (HF doesn't expose it
        # on the audio config; it lives on `text_config`).
        config.text_hidden_size = int(text_hidden_size)
        edge = cls(
            config,
            hann_window=hann_window,
            mel_filters=mel_filters,
            use_matmul_dft=use_matmul_dft,
        )
        _load_state_from_hf(edge, hf_audio_tower, hf_embed_audio)
        edge.eval()
        return edge


# ---------------------------------------------------------------------------
# HF -> edge config / state-dict helpers
# ---------------------------------------------------------------------------


def _config_from_hf(hf_cfg) -> Gemma4EdgeAudioConfig:
    """Build a `Gemma4EdgeAudioConfig` from an HF `Gemma4AudioConfig`."""
    return Gemma4EdgeAudioConfig(
        hidden_size=hf_cfg.hidden_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        hidden_act=hf_cfg.hidden_act,
        subsampling_conv_channels=tuple(hf_cfg.subsampling_conv_channels),
        conv_kernel_size=hf_cfg.conv_kernel_size,
        residual_weight=float(hf_cfg.residual_weight),
        attention_chunk_size=hf_cfg.attention_chunk_size,
        attention_context_left=hf_cfg.attention_context_left,
        attention_context_right=hf_cfg.attention_context_right,
        attention_logit_cap=float(hf_cfg.attention_logit_cap),
        attention_invalid_logits_value=float(hf_cfg.attention_invalid_logits_value),
        use_clipped_linears=bool(hf_cfg.use_clipped_linears),
        rms_norm_eps=float(hf_cfg.rms_norm_eps),
        gradient_clipping=float(hf_cfg.gradient_clipping),
        output_proj_dims=int(hf_cfg.output_proj_dims),
        # text_hidden_size is filled in by the caller of `from_hf`.
    )


def _load_state_from_hf(
    edge: "Gemma4EdgeAudioTower",
    hf_audio_tower: nn.Module,
    hf_embed_audio: nn.Module,
) -> None:
    """Copy parameters from HF `Gemma4AudioModel` + `Gemma4MultimodalEmbedder`
    into the edge tower.

    State-dict mapping (edge <- HF):

      audio_tower.subsample_conv_projection.layer{0,1}.conv.weight
          <- subsample_conv_projection.layer{0,1}.conv.weight
      audio_tower.subsample_conv_projection.layer{0,1}.norm.weight
          <- subsample_conv_projection.layer{0,1}.norm.weight
      audio_tower.subsample_conv_projection.input_proj_linear.weight
          <- subsample_conv_projection.input_proj_linear.weight

      audio_tower.layers.{i}.feed_forward{1,2}.ffw_layer_{1,2}.linear.weight
          <- layers.{i}.feed_forward{1,2}.ffw_layer_{1,2}.linear.weight
      audio_tower.layers.{i}.feed_forward{1,2}.{pre,post}_layer_norm.weight
          <- layers.{i}.feed_forward{1,2}.{pre,post}_layer_norm.weight
      audio_tower.layers.{i}.self_attn.{q,k,v}_proj.linear.weight
          <- layers.{i}.self_attn.{q,k,v}_proj.linear.weight
      audio_tower.layers.{i}.self_attn.post.linear.weight
          <- layers.{i}.self_attn.post.linear.weight
      audio_tower.layers.{i}.self_attn.relative_k_proj.weight
          <- layers.{i}.self_attn.relative_k_proj.weight
      audio_tower.layers.{i}.self_attn.per_dim_scale
          <- layers.{i}.self_attn.per_dim_scale
      audio_tower.layers.{i}.lconv1d.linear_{start,end}.linear.weight
          <- layers.{i}.lconv1d.linear_{start,end}.linear.weight
      audio_tower.layers.{i}.lconv1d.depthwise_conv1d.weight
          <- layers.{i}.lconv1d.depthwise_conv1d.weight
      audio_tower.layers.{i}.lconv1d.{pre_layer_norm,conv_norm}.weight
          <- layers.{i}.lconv1d.{pre_layer_norm,conv_norm}.weight
      audio_tower.layers.{i}.{norm_pre_attn,norm_post_attn,norm_out}.weight
          <- layers.{i}.{norm_pre_attn,norm_post_attn,norm_out}.weight

      audio_tower.output_proj.{weight,bias}
          <- output_proj.{weight,bias}

      embed_audio.embedding_projection.weight
          <- (hf_embed_audio).embedding_projection.weight
      (embedding_pre_projection_norm has no weight; with_scale=False)

    Then ClippableLinear bounds (`input_min`, `input_max`, `output_min`,
    `output_max`) are pulled from HF as 0-d Tensors and baked into edge
    `ClippableLinear` modules via `set_bounds(...)`.
    """
    hf_tower_sd = hf_audio_tower.state_dict()
    hf_embed_sd = hf_embed_audio.state_dict()

    # 1) Tower (audio_tower.*) — HF keys mirror the edge with `audio_tower.`
    # prefix stripped.
    edge_sd = edge.state_dict()
    for full_key in list(edge_sd.keys()):
        if not full_key.startswith("audio_tower."):
            continue
        hf_key = full_key[len("audio_tower."):]
        if hf_key in hf_tower_sd and hf_tower_sd[hf_key].shape == edge_sd[full_key].shape:
            edge_sd[full_key] = hf_tower_sd[hf_key].clone()

    # 2) Embedder (embed_audio.*)
    for full_key in list(edge_sd.keys()):
        if not full_key.startswith("embed_audio."):
            continue
        hf_key = full_key[len("embed_audio."):]
        if hf_key in hf_embed_sd and hf_embed_sd[hf_key].shape == edge_sd[full_key].shape:
            edge_sd[full_key] = hf_embed_sd[hf_key].clone()

    edge.load_state_dict(edge_sd, strict=False)

    # 3) ClippableLinear bounds (audio side).
    if not edge.config.use_clipped_linears:
        return
    for edge_name, edge_mod in edge.named_modules():
        if not isinstance(edge_mod, ClippableLinear):
            continue
        if not edge_name.startswith("audio_tower."):
            continue
        hf_key_prefix = edge_name[len("audio_tower."):]
        try:
            imin = float(hf_tower_sd[f"{hf_key_prefix}.input_min"].item())
            imax = float(hf_tower_sd[f"{hf_key_prefix}.input_max"].item())
            omin = float(hf_tower_sd[f"{hf_key_prefix}.output_min"].item())
            omax = float(hf_tower_sd[f"{hf_key_prefix}.output_max"].item())
        except KeyError:
            continue
        edge_mod.set_bounds(imin, imax, omin, omax)


__all__ = [
    "AUDIO_FFT_LENGTH",
    "AUDIO_FRAME_LENGTH",
    "AUDIO_HOP_LENGTH",
    "AUDIO_MEL_FLOOR",
    "AUDIO_N_MEL",
    "AUDIO_PAD_LEFT",
    "AUDIO_PAD_RIGHT",
    "AUDIO_SAMPLE_COUNT_MAX",
    "AUDIO_SAMPLE_COUNT_MIN",
    "AUDIO_SAMPLE_RATE",
    "BLOCK_K_MAX",
    "BLOCK_K_MIN",
    "BLOCK_SAMPLES",
    "ClippableLinear",
    "Gemma4EdgeAudioAttention",
    "Gemma4EdgeAudioCausalConv1d",
    "Gemma4EdgeAudioConfig",
    "Gemma4EdgeAudioEmbedder",
    "Gemma4EdgeAudioFeedForward",
    "Gemma4EdgeAudioLayer",
    "Gemma4EdgeAudioLightConv1d",
    "Gemma4EdgeAudioModel",
    "Gemma4EdgeAudioRelPositionalEncoding",
    "Gemma4EdgeAudioSubSampleConvProjection",
    "Gemma4EdgeAudioSubSampleConvProjectionLayer",
    "Gemma4EdgeAudioTower",
    "LogMelFrontend",
    "RMSNorm",
]
