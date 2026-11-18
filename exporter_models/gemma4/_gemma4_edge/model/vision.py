"""
Gemma 4 — Vision Encoder (ExecuTorch edge implementation)
=========================================================

Faithfully matches the reference transformers implementation
(transformers/models/gemma4/modeling_gemma4.py, lines 540-1014).

Architecture:
  1. Patch embedder: linear projection of flattened patches + learned 2D
     position embedding (x + y) gated by a padding mask.
  2. Encoder: stack of N Gemma4VisionEncoderLayer blocks. Each layer is a
     standard pre-norm Transformer block with QKV-norm and a 2D rotary
     embedding ("apply_multidimensional_rope": rotation applied independently
     to the two halves of head_dim with (x, y) position frequencies).
  3. Pooler: zero-out padding rows then optional 2D average pooling over a
     `k×k` grid to compress N patches into `MAX_SOFT_TOKENS` soft tokens,
     followed by multiplication with `sqrt(hidden_size)`.
  4. Optional `(x - std_bias) * std_scale` post-standardization.

Edge-friendly simplifications relative to HF (math preserved):
  - Plain `nn.Module` everywhere; no `GradientCheckpointingLayer`, no
    `**kwargs: Unpack[…]`, no `ALL_ATTENTION_FUNCTIONS` lookup. The single
    attention path is `eager_attention_forward`-equivalent: matmul, scale,
    additive mask, softmax (upcast to fp32), matmul.
  - Rotary embedding pre-bakes the per-coordinate cos/sin lookup tables in
    `__init__` (analogous to the text-decoder precedent). The forward path
    is then a single `embedding` gather over the integer position ids, two
    concatenations, and one half-axis split — much smaller shader fan-out
    than HF's per-call `inv_freq @ position_ids` matmul + cat. Math is
    bit-equivalent (both use default rope_type with full-fp32 evaluation).
  - `Gemma4ClippableLinear` is collapsed to a plain `nn.Linear` plus a
    pair of Python-scalar `torch.clamp` calls (compile-time scalars in the
    exported graph). The HF version registers 0-d Tensor buffers per
    projection — that triggers `aten.clamp.Tensor` which Vulkan does not
    register; the scalar form lowers to `aten.clamp.default`.
  - `create_bidirectional_mask` is reimplemented inline (no recursion into
    the transformers masking_utils, sdpa/eager dispatch, vmap context, or
    `_preprocess_mask_arguments`): a flat 4D boolean-then-float mask of
    shape `(B, 1, S, S)` derived from the 1D padding vector. The factory
    function is purely bidirectional (`q_idx >= 0`), so it reduces to a
    `kv_idx`-only padding test that broadcasts over `q_idx`.
  - One-hot @ embedding_table in HF's `_position_embeddings` is left as-is
    (math-equivalent to `nn.Embedding` lookup; one_hot allows
    `pixel_position_ids == -1` rows to mask cleanly via clamp+where) but
    the clamp-where pair is fused into a single mask call.
  - All linear projections use `nn.Linear(..., bias=False)`. Tensor `.view`
    over `.reshape` where layout is contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Gemma4EdgeVisionConfig:
    """Config for Gemma 4 vision tower, mirroring `Gemma4VisionConfig`.

    Only fields actually consumed by the edge implementation are listed.
    Fields like `base_model_tp_plan`, `default_theta`, `initializer_range`
    are HF-only and intentionally omitted.
    """

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6

    # patch / position embedding
    patch_size: int = 16
    position_embedding_size: int = 10 * 1024

    # rotary
    rope_theta: float = 100.0
    max_position_embeddings: int = 131_072

    # pooler
    pooling_kernel_size: int = 3

    # behavioural flags (Gemma4ClippableLinear, optional post-standardize)
    use_clipped_linears: bool = False
    standardize: bool = False

    # held only for parity with HF surface; not consumed in forward
    attention_dropout: float = 0.0

    # ---------------- Stage B simplification flags (opt-in) ----------------
    # B1 — One-hot @ embedding_table -> two `F.embedding` lookups summed.
    #      Patch embedder's `_position_embeddings` replaces the
    #      `(B, P, 2, P_max) @ (2, P_max, H)` matmul with two per-axis gathers.
    #      Mathematically equivalent (reduces a 23 MB ephemeral tensor +
    #      a wide matmul shader pipeline to two embedding gathers + add).
    use_embedding_position_emb: bool = False
    # B2 — Pooler one-hot matmul -> reshape + mean.
    #      `_avg_pool_by_positions` rewrites the 2D spatial pool as a pure
    #      reshape + mean over the within-block axes, dropping the one-hot
    #      construction and (B, num_patches, length) matmul shader.
    #      Requires patches laid out in row-major spatial order with fixed
    #      pool stride; the production export pipeline always uses this.
    #      The grid dimensions (`patch_grid_nx`, `patch_grid_ny`) MUST be
    #      set on the config so the reshape is fully static at export time
    #      (avoids a `.max().item()` data-dependent guard in torch.export).
    use_reshape_pool: bool = False
    # Static grid shape used when `use_reshape_pool=True`. These are filled
    # by the caller before constructing the tower (the shipping recipes use
    # VIS_NX=40, VIS_NY=28 at 448x640). When the flag is False these are
    # unused.
    patch_grid_nx: int = 0
    patch_grid_ny: int = 0
    # B3 — Pre-baked static bidirectional mask.
    #      Vision tower at inference always processes a full 448x640 image
    #      with zero padded patches, so the bidirectional float mask is a
    #      constant `zeros(1, 1, kv_len, kv_len)`. Pre-bake as a buffer in
    #      `Gemma4EdgeVisionEncoder.__init__`; skip mask construction in
    #      forward. Flag also tells the encoder to IGNORE the runtime
    #      `attention_mask` arg (every call site passes the all-True
    #      padding-derived mask anyway).
    use_static_mask: bool = False
    # B4 — Fused SDPA via `F.scaled_dot_product_attention`.
    #      Replace the explicit Q @ K^T -> mask-add -> softmax -> @ V chain
    #      with a single `F.scaled_dot_product_attention` call. Lowers to
    #      `aten.scaled_dot_product_attention` which the Vulkan partitioner
    #      may either fuse as one delegate or decompose into the same set
    #      of ops (in which case the simplification is a no-op).
    #      WARNING: on Vulkan this is a FOOTGUN — see
    #      `project_gemma4_F_sdpa_decomp_arena_bomb.md`. Use B5 instead.
    use_fused_sdpa: bool = False
    # B5 — Single-op SDPA via `et_vk::gemma_sdpa_with_kv_cache`.
    #      Replaces the explicit Q @ K^T -> softmax -> @ V chain with one
    #      Vulkan SDPA call. Layout is BSHD (no `.transpose(1,2)` after RoPE).
    #      Requires:
    #        - `use_static_mask=True` (B3): the SDPA op consumes the same
    #          baked `(1, 1, S, S)` zeros mask; passing it bypasses the
    #          shader's causal short-circuit (verified at GemmaSDPA.cpp:345
    #          and gemma_sdpa_compute_attn_weights_tiled.glsl HAS_MASK branch).
    #        - `bake_sdpa_caches(num_patches, dtype)` to materialize per-layer
    #          `(1, num_patches, num_kv_heads, head_dim)` scratch buffers
    #          BEFORE export.
    #      Each layer currently allocates its own scratch K/V cache pair
    #      (~+7 MB per layer in fp32). Step B (realize_cache_once dedup in
    #      GemmaSDPA.cpp) will let all layers share one pair on the encoder.
    use_vulkan_sdpa: bool = False


# ---------------------------------------------------------------------------
# RMSNorm — identical to text-decoder precedent
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm matching `Gemma4RMSNorm` (with optional learnable scale)."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        self.dim = dim
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = (x * x).mean(-1, keepdim=True) + self.eps
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
    """Edge replacement for `Gemma4ClippableLinear`.

    The HF version registers four 0-d buffers per projection
    (`input_min`/`input_max`/`output_min`/`output_max`). Vulkan's op
    registry only knows `aten.clamp.default` (Python-scalar bounds), not
    `aten.clamp.Tensor` (0-d buffer bounds). We bake the bounds into
    Python floats at `__init__` time; if `use_clipped_linears=False` the
    forward degenerates to `self.linear(x)`.

    Math is bit-exact relative to HF's `Gemma4ClippableLinear` when the
    bounds are loaded from the original `input_min` etc. buffers.
    """

    def __init__(
        self,
        config: Gemma4EdgeVisionConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # Bounds are filled by the HF state-dict loader; default to identity.
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
# Patch embedder
# ---------------------------------------------------------------------------

class Gemma4EdgeVisionPatchEmbedder(nn.Module):
    """Mirror of `Gemma4VisionPatchEmbedder` (HF lines 540-572).

    Inputs:
      - `pixel_values`: `(B, num_patches, 3 * patch_size^2)` already
        unfolded by caller (HF processor's patchify).
      - `pixel_position_ids`: `(B, num_patches, 2)` int64.
      - `padding_positions`: `(B, num_patches)` bool, True = padding.

    Output: `(B, num_patches, hidden_size)` ready for the encoder.

    Internal `_position_embeddings` follows HF: one-hot of x and y
    positions over the position_embedding_table (2, P_max, H), matmul,
    sum over the {x, y} axis. The clamp(min=0) handles `-1`-marked
    padding indices; the result is masked back to 0 via `torch.where`.
    """

    def __init__(self, config: Gemma4EdgeVisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size

        self.input_proj = nn.Linear(
            3 * self.patch_size * self.patch_size, self.hidden_size, bias=False
        )
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )
        # B1: pre-split position table into two contiguous parameters.
        # Loader fills them; forward picks the per-axis slice. Keeping
        # the original `position_embedding_table` for state-dict compat
        # (the loader populates both forms).
        if getattr(config, "use_embedding_position_emb", False):
            self.position_embedding_x = nn.Parameter(
                torch.ones(self.position_embedding_size, self.hidden_size)
            )
            self.position_embedding_y = nn.Parameter(
                torch.ones(self.position_embedding_size, self.hidden_size)
            )

    def _position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        clamped = pixel_position_ids.clamp(min=0)
        if getattr(self.config, "use_embedding_position_emb", False):
            # B1: two per-axis `F.embedding` lookups summed.
            # Mathematically: out[b,p,h] =
            #     table[0, pos[b,p,0], h] + table[1, pos[b,p,1], h]
            # which is exactly two embedding lookups on slices of the
            # full table. Use pre-split `position_embedding_x`/`_y`
            # parameters so Vulkan partitioner sees true constant weights
            # (slicing the combined table at trace time produces a non-
            # constant value that fails val_is_tref at prepack).
            ids_x = clamped[..., 0]  # (B, P)
            ids_y = clamped[..., 1]  # (B, P)
            emb_x = F.embedding(ids_x, self.position_embedding_x)
            emb_y = F.embedding(ids_y, self.position_embedding_y)
            position_embeddings = emb_x + emb_y
        else:
            # (B, num_patches, 2, P_max) one-hot, then permute to
            # (B, 2, P, P_max).
            one_hot = F.one_hot(
                clamped, num_classes=self.position_embedding_size
            )
            one_hot = one_hot.permute(0, 2, 1, 3).to(
                self.position_embedding_table
            )
            # (B, 2, P, P_max) @ (2, P_max, H) -> (B, 2, P, H); sum over 2.
            position_embeddings = one_hot @ self.position_embedding_table
            position_embeddings = position_embeddings.sum(dim=1)
        # Zero out embeddings for padding patches.
        position_embeddings = torch.where(
            padding_positions.unsqueeze(-1), 0.0, position_embeddings
        )
        return position_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        # Gemma4 applies no normalization and instead scales in model code.
        pixel_values = 2.0 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(
            pixel_position_ids, padding_positions
        )
        return hidden_states + position_embeddings


# ---------------------------------------------------------------------------
# Rotary embedding (2D, pre-baked)
# ---------------------------------------------------------------------------

class Gemma4EdgeVisionRotaryEmbedding(nn.Module):
    """Mirror of `Gemma4VisionRotaryEmbedding` + `apply_multidimensional_rope`.

    HF computes inv_freq on the fly per forward; we pre-bake a
    `(num_positions, head_dim // 2)` lookup table once. Forward gathers
    per-axis cos/sin via plain integer indexing, concatenates to
    `(B, S, head_dim)` and returns the (cos, sin) pair.

    The math is bit-equivalent to HF for `rope_type="default"` because
    the per-position frequency is `inv_freq * position` (an integer
    multiply, evaluated in fp32 in HF and here).
    """

    def __init__(self, config: Gemma4EdgeVisionConfig):
        super().__init__()
        self.config = config
        # head_dim split into 2 spatial halves; each half is rotated.
        head_dim = config.head_dim
        self.head_dim = head_dim
        self.ndim = 2
        # Per HF's compute_default_rope_parameters: spatial_dim = head_dim // 2,
        # then inv_freq is on `arange(0, spatial_dim, 2) / spatial_dim`.
        spatial_dim = head_dim // 2
        self.spatial_dim = spatial_dim
        self.num_rotated_per_dim = 2 * (head_dim // (2 * self.ndim))  # = head_dim // ndim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.int64).float()
                / spatial_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-bake cos/sin tables for positions [0, max_positions).
        # Position ids are bounded by the patch grid size; cap generously.
        # The HF default-rope dynamic update only kicks in when seq_len
        # exceeds `max_seq_len_cached`, which is `max_position_embeddings`;
        # the grid is far smaller (max 40 here) so the default table is
        # used unconditionally.
        max_positions = config.position_embedding_size
        t = torch.arange(max_positions, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (P, spatial_dim/2)
        # HF cat (freqs, freqs) along last dim -> (P, spatial_dim) per axis.
        emb_per_axis = torch.cat([freqs, freqs], dim=-1)  # (P, spatial_dim)
        self.register_buffer("cos_table", emb_per_axis.cos(), persistent=False)
        self.register_buffer("sin_table", emb_per_axis.sin(), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) of shape `(B, num_patches, head_dim)`.

        Each axis contributes a `(B, num_patches, spatial_dim)` block; the
        two blocks are concatenated along the last dim.
        """
        # position_ids: (B, num_patches, 2). Negative => padding; clamp to 0
        # for the lookup. HF clamp/where mask happens upstream of attention.
        ids = position_ids.clamp(min=0).long()  # (B, P, 2)
        # cos/sin per axis: gather rows from the pre-baked table.
        cos_x = self.cos_table[ids[..., 0]]  # (B, P, spatial_dim)
        sin_x = self.sin_table[ids[..., 0]]
        cos_y = self.cos_table[ids[..., 1]]
        sin_y = self.sin_table[ids[..., 1]]
        cos = torch.cat([cos_x, cos_y], dim=-1).to(hidden_states.dtype)
        sin = torch.cat([sin_x, sin_y], dim=-1).to(hidden_states.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim to the front, negate it.

    For x of last-dim D, returns `(-x[..., D/2:], x[..., :D/2])` along dim -1.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_multidim_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotated_per_dim: int,
) -> torch.Tensor:
    """Edge equivalent of `apply_multidimensional_rope` (HF lines 803-856).

    x:    `(B, num_patches, num_heads, head_dim)`
    cos:  `(B, num_patches, head_dim)`        (per-axis rotary blocks concatenated)
    sin:  same shape as cos
    rotated_per_dim: HF's `num_rotated_channels_per_dim`. For head_dim=64,
                     ndim=2: rotated_per_dim = 32 (two 32-wide axis blocks).

    Math: HF splits x and cos/sin into `ndim=2` per-axis chunks of width
    `rotated_per_dim` along the last dim, applies
    `(chunk * cos_chunk) + (rotate_half(chunk) * sin_chunk)` per chunk, then
    concatenates the two chunks back. CRUCIALLY, `rotate_half` operates
    WITHIN each chunk (splitting the chunk into halves of `rotated_per_dim/2`).
    A naive single-call `rotate_half` over the full head_dim would split
    into halves of `head_dim/2 = rotated_per_dim` and produce a different
    interleaving, so we must apply per-chunk rotation explicitly.
    """
    cos = cos.unsqueeze(2)  # (B, P, 1, head_dim)
    sin = sin.unsqueeze(2)
    # Split into 2 per-axis chunks along the last dim (assumes ndim=2).
    x_chunks = torch.split(x, rotated_per_dim, dim=-1)
    cos_chunks = torch.split(cos, rotated_per_dim, dim=-1)
    sin_chunks = torch.split(sin, rotated_per_dim, dim=-1)
    out_chunks = []
    for xi, ci, si in zip(x_chunks, cos_chunks, sin_chunks):
        out_chunks.append((xi * ci) + (_rotate_half(xi) * si))
    out = torch.cat(out_chunks, dim=-1)
    return out.to(x.dtype)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Gemma4EdgeVisionAttention(nn.Module):
    """Mirror of `Gemma4VisionAttention` (HF lines 860-927).

    Explicit eager forward: matmul, additive mask, softmax (fp32), matmul.
    GQA is supported in principle (num_attention_heads vs num_key_value_heads
    may differ), but the e2b checkpoint sets them equal so `kv_groups == 1`.
    """

    def __init__(self, config: Gemma4EdgeVisionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        # HF sets scaling = 1.0 because QKV-norm pre-normalizes the activations.
        self.scaling: float = 1.0
        self.q_proj = ClippableLinear(
            config, config.hidden_size, self.num_heads * self.head_dim
        )
        self.k_proj = ClippableLinear(
            config, config.hidden_size, self.num_kv_heads * self.head_dim
        )
        self.v_proj = ClippableLinear(
            config, config.hidden_size, self.num_kv_heads * self.head_dim
        )
        self.o_proj = ClippableLinear(
            config, self.num_heads * self.head_dim, config.hidden_size
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
        # `rotated_per_dim` parameter for apply_multidim_rope.
        self.rotated_per_dim = 2 * (self.head_dim // (2 * 2))  # ndim=2
        # B5 deliberately does NOT register K/V scratch buffers. The Vulkan
        # SDPA op's `Tensor(a!)` mutation annotation, combined with the
        # vision encoder's pooler-forced partition fragmentation, drives
        # `_verify_exported_program_signature` into a positional-slice
        # ordering bug where non-mutation outputs (e.g. ClippableLinear
        # clamps) land in the mutation-output position range and fail the
        # `mutation_node in gs.buffers_to_mutate` check. Ephemeral caches
        # allocated inside `forward()` route the mutation through
        # `auto_functionalized_v2` as a local clone+modify with no external
        # buffer tracking; `gs.buffers_to_mutate` stays empty for vision.

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        cos, sin = position_embeddings

        use_vulkan_sdpa = getattr(self.config, "use_vulkan_sdpa", False)

        # Project, view to per-head shape, apply qkv-norm, apply 2D rotary.
        # B5 keeps the BSHD layout (no `.transpose(1, 2)` after RoPE); the
        # Vulkan SDPA op consumes (B, S, H, D) directly.
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_multidim_rope(q, cos, sin, self.rotated_per_dim)

        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = apply_multidim_rope(k, cos, sin, self.rotated_per_dim)

        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_norm(v)

        if use_vulkan_sdpa:
            if not _HAS_VULKAN_SDPA_OPS:
                raise RuntimeError(
                    "use_vulkan_sdpa=True requires gemma4.model.custom_sdpa_ops "
                    "to be importable. Check that gemma4 is on PYTHONPATH."
                )
            # B5: single et_vk.gemma_custom_sdpa call (read-only variant —
            # no Tensor(a!) mutation, no auto_functionalized, no clone
            # insertion). The schema is
            #   gemma_custom_sdpa(query, key_cache, value_cache,
            #                     start_pos, seq_len, attn_mask, is_causal)
            # By contract the cache "already contains" the K/V to attend
            # against. For vision attention, K/V have the exact shape of
            # the cache (1, num_patches, num_kv_heads, head_dim), so we
            # pass `k` and `v` DIRECTLY as the cache arguments.
            #
            # The C++ side relies on `realize_cache_once` (in
            # backends/vulkan/runtime/graph/ops/impl/GemmaSDPA.cpp) to
            # bypass `add_tensor_like` when cache_data is already a graph
            # Tensor — returning k/v directly. (See gemma_custom_sdpa_impl
            # — this is the realize-cache-once edit shipped alongside
            # this B5 path.)
            #
            # Mask path bypasses causal masking (GemmaSDPA.cpp:345-355 +
            # gemma_sdpa_compute_attn_weights_tiled.glsl HAS_MASK branch);
            # we pass the B3 static (1,1,S,S) zeros mask so attention is
            # fully bidirectional.
            #
            # Layout: BSHD (no `.transpose(1,2)` after RoPE). GQA broadcast
            # is handled inside the shader.
            attn_output = torch.ops.et_vk.gemma_custom_sdpa.default(
                q,
                k, v,        # key_cache=k, value_cache=v (direct pass-through)
                0,           # start_pos
                S,           # seq_len
                attention_mask,
                False,       # is_causal (ignored when attn_mask is provided)
            )
            # Output shape: (B, S, H_q, D) -> (B, S, H_q * D)
            attn_output = attn_output.contiguous().view(B, S, -1)
        else:
            q = q.transpose(1, 2)  # (B, H, S, D)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # GQA broadcast (only kicks in if num_kv_groups > 1).
            if self.num_key_value_groups != 1:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)

            if getattr(self.config, "use_fused_sdpa", False):
                # B4: single F.scaled_dot_product_attention call. FOOTGUN on
                # Vulkan (see project_gemma4_F_sdpa_decomp_arena_bomb.md).
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.scaling,
                )
            else:
                # attn_weights = (Q @ K^T) * scale
                attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                # softmax in fp32, then cast back.
                attn_weights = F.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v)
            # (B, H, S, D) -> (B, S, H*D)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Gemma4EdgeVisionMLP(nn.Module):
    """Mirror of `Gemma4VisionMLP` (HF lines 633-647).

    GeGLU FFN with GELU(tanh) activation:
        down_proj( gelu(gate_proj(x)) * up_proj(x) ).
    """

    def __init__(self, config: Gemma4EdgeVisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = ClippableLinear(config, self.hidden_size, self.intermediate_size)
        self.up_proj = ClippableLinear(config, self.hidden_size, self.intermediate_size)
        self.down_proj = ClippableLinear(config, self.intermediate_size, self.hidden_size)
        if config.hidden_activation == "gelu_pytorch_tanh":
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        elif config.hidden_activation == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------

class Gemma4EdgeVisionEncoderLayer(nn.Module):
    """Mirror of `Gemma4VisionEncoderLayer` (HF lines 929-970).

    Plain `nn.Module` (NOT `GradientCheckpointingLayer`). Sandwich norms:
    pre-norm + post-norm around both attention and FFN.
    """

    def __init__(self, config: Gemma4EdgeVisionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4EdgeVisionAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4EdgeVisionMLP(config)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # --- Attention block ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        # --- FFN block ---
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@dataclass
class _EncoderOutput:
    """Lightweight stand-in for `BaseModelOutputWithPast.last_hidden_state`.

    `VisionEncoderExport` reads `enc_out.last_hidden_state` only;
    keeping the wrapper minimal avoids dragging in HF model_outputs.
    """
    last_hidden_state: torch.Tensor


def _build_bidirectional_attention_mask(
    attention_mask_2d: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Inline replacement for `create_bidirectional_mask` (HF).

    HF traces the following path for our use case:
        bidirectional_mask_function: q_idx >= 0  (always True)
        and padding_mask_function:    padding_mask[batch_idx, kv_idx]

    The "and" mask reduces to "kv_idx is valid". HF then builds a
    4D float mask of shape (B, 1, q_length, kv_length) where True->0.0
    and False->finfo(dtype).min (eager path).

    Here we accept a `(B, kv_length)` bool `attention_mask_2d` (True=valid,
    False=padding) and produce a `(B, 1, q_length, kv_length)` additive
    float mask suitable for adding to `Q@K^T` pre-softmax.
    """
    B, kv_len = attention_mask_2d.shape
    # broadcast (B, 1, 1, kv_len) over q_len at the call site (mask is
    # padding-only, so it does not depend on q_idx).
    bool_mask = attention_mask_2d[:, None, None, :]  # (B, 1, 1, kv_len)
    min_val = torch.finfo(dtype).min
    additive = torch.where(
        bool_mask,
        torch.tensor(0.0, dtype=dtype, device=attention_mask_2d.device),
        torch.tensor(min_val, dtype=dtype, device=attention_mask_2d.device),
    )
    return additive


class Gemma4EdgeVisionEncoder(nn.Module):
    """Mirror of `Gemma4VisionEncoder` (HF lines 973-1014)."""

    def __init__(self, config: Gemma4EdgeVisionConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.rotary_emb = Gemma4EdgeVisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [
                Gemma4EdgeVisionEncoderLayer(config=config, layer_idx=i)
                for i in range(self.num_layers)
            ]
        )
        # B3: When `use_static_mask` is enabled and `use_fused_sdpa` is NOT,
        # pre-bake the (1, 1, S, S) additive float mask as a constant buffer
        # (zeros because vision inference has no padded patches).
        # Buffer is registered with `persistent=True` so the Vulkan prepack
        # path sees it as a true TensorRef (val_is_tref check). Non-persistent
        # buffers can get lifted to runtime inputs in some export flows which
        # then fail the val_is_tref guard at Staging.cpp:207.
        if getattr(config, "use_static_mask", False) and not getattr(
            config, "use_fused_sdpa", False
        ):
            # Buffer length matches the maximum patch count (= num_patches at
            # inference). The caller binds it via `bake_static_mask` before
            # export; default of `zeros(0)` is a placeholder.
            self.register_buffer(
                "_static_attention_mask",
                torch.zeros(0, dtype=torch.float32),
                persistent=True,
            )
        # When `use_fused_sdpa` is True, the F.scaled_dot_product_attention
        # call accepts an (S, S) bool mask directly; we skip baking the
        # additive form.

    def bake_static_mask(
        self, num_patches: int, dtype: torch.dtype = torch.float32
    ) -> None:
        """Materialize the (1, 1, num_patches, num_patches) all-zeros
        additive mask used when `use_static_mask=True`.

        Should be called after `__init__` but before export, once the
        caller knows the static patch count.
        """
        if not getattr(self.config, "use_static_mask", False):
            return
        if getattr(self.config, "use_fused_sdpa", False):
            return
        # Padding is always all-False at inference: bidirectional float mask
        # collapses to zeros everywhere.
        self._static_attention_mask = torch.zeros(
            1, 1, num_patches, num_patches, dtype=dtype
        )

    def bake_sdpa_caches(
        self, num_patches: int, dtype: torch.dtype = torch.float32
    ) -> None:
        """Compat shim — kept so existing export scripts still call cleanly.

        Earlier B5 designs pre-baked per-layer K/V scratch buffers; the
        current shape allocates them ephemerally inside `Attention.forward`
        to side-step a partitioner+verifier ordering bug (see comment in
        `Gemma4EdgeVisionAttention.__init__`). No state to materialize here.
        """
        del num_patches, dtype  # accepted for API parity, unused

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> _EncoderOutput:
        """Returns an object with `.last_hidden_state` attribute.

        Args:
          inputs_embeds:        `(B, num_patches, hidden_size)`.
          attention_mask:       `(B, num_patches)` bool, True == valid.
          pixel_position_ids:   `(B, num_patches, 2)` int64. -1 == padding.
        """
        if getattr(self.config, "use_static_mask", False):
            if getattr(self.config, "use_fused_sdpa", False):
                # SDPA path needs no additive mask (caller sets is_causal=False
                # and passes attn_mask=None for the no-padding inference case).
                attention_mask_4d = None
            else:
                # Use the pre-baked all-zeros additive mask.
                attention_mask_4d = self._static_attention_mask.to(
                    inputs_embeds.dtype
                )
        else:
            attention_mask_4d = _build_bidirectional_attention_mask(
                attention_mask, dtype=inputs_embeds.dtype
            )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask_4d,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
            )
        return _EncoderOutput(last_hidden_state=hidden_states)


# ---------------------------------------------------------------------------
# Pooler
# ---------------------------------------------------------------------------

class Gemma4EdgeVisionPooler(nn.Module):
    """Mirror of `Gemma4VisionPooler` (HF lines 574-630).

    Exposes the same attributes `VisionEncoderExport` uses:
      - `root_hidden_size` (float, sqrt of hidden_size);
      - `_avg_pool_by_positions(hidden, pos_ids, length)` returning
        `(pooled, mask)`.
    """

    def __init__(self, config: Gemma4EdgeVisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.root_hidden_size = float(self.hidden_size) ** 0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """2D spatial average pooling: collapses N=length*k^2 patches to
        length soft tokens by averaging over k×k patch grids."""
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k * k
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: k={k}^2 "
                f"times length={length} must be {input_seq_len}."
            )
        if getattr(self.config, "use_reshape_pool", False):
            # B2: spatial average pool via aten.avg_pool2d.default.
            # Patches are laid out row-major over a (ny, nx) static grid;
            # grid dims come from config (set by caller before construction).
            # Reshape `(B, ny*nx, H)` -> `(B, H, ny, nx)` (channels-first),
            # call `F.avg_pool2d(..., kernel_size=k, stride=k)`, then reshape
            # back. Vulkan implements `aten.avg_pool2d.default`; a 6D mean
            # over (B, sy, k, sx, k, H) was rejected by the partitioner.
            B = hidden_states.shape[0]
            H = hidden_states.shape[2]
            nx = int(self.config.patch_grid_nx)
            ny = int(self.config.patch_grid_ny)
            assert nx > 0 and ny > 0, (
                "use_reshape_pool requires config.patch_grid_nx / "
                "patch_grid_ny to be set before construction."
            )
            assert nx * ny == input_seq_len, (
                f"reshape_pool grid mismatch: nx*ny={nx*ny} but "
                f"input_seq_len={input_seq_len}"
            )
            assert nx % k == 0 and ny % k == 0, (
                f"reshape_pool requires nx,ny divisible by k={k}; "
                f"got nx={nx}, ny={ny}"
            )
            sx_count = nx // k
            sy_count = ny // k
            # (B, ny*nx, H) -> (B, ny, nx, H) -> (B, H, ny, nx)
            x = hidden_states.view(B, ny, nx, H).permute(0, 3, 1, 2).contiguous()
            pooled_chw = F.avg_pool2d(x, kernel_size=k, stride=k)
            # (B, H, sy, sx) -> (B, sy, sx, H) -> (B, sy*sx, H)
            output = pooled_chw.permute(0, 2, 3, 1).contiguous().view(
                B, sy_count * sx_count, H
            )
            # Mask: a soft-token position is valid iff at least one source
            # patch is non-padding. At inference, padding is always all-False,
            # so the mask is all-True. Build the test from the original
            # `pixel_position_ids` (NOT clamped) to mirror the one-hot
            # semantics: pos == -1 produces a zero one-hot row.
            valid_patch = (pixel_position_ids[..., 0] >= 0)  # (B, P)
            valid = valid_patch.view(B, sy_count, k, sx_count, k)
            mask = valid.any(dim=2).any(dim=3)            # (B, sy, sx)
            mask = mask.view(B, sy_count * sx_count)
            return output.to(hidden_states.dtype), mask
        clamped = pixel_position_ids.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        # one_hot to (B, num_patches, length), normalize by k^2, transpose
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                "Cannot output more soft tokens"
                f" (requested {output_length}) than there are patches"
                f" ({hidden_states.shape[1]})."
            )
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, padding_positions


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------

class Gemma4EdgeVisionTower(nn.Module):
    """Top-level wrapper exposing the surface `VisionEncoderExport` reads.

    Attributes:
      - `.config`           — Gemma4EdgeVisionConfig
      - `.patch_embedder`   — Gemma4EdgeVisionPatchEmbedder
      - `.encoder`          — Gemma4EdgeVisionEncoder
      - `.pooler`           — Gemma4EdgeVisionPooler
      - `.std_bias` / `.std_scale` — registered buffers (only if standardize)

    Forward is provided for parity-testing convenience; the export wrapper
    calls the sub-modules directly via `_soft_tokens()`.
    """

    def __init__(self, config: Gemma4EdgeVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4EdgeVisionPatchEmbedder(config)
        self.encoder = Gemma4EdgeVisionEncoder(config)
        self.pooler = Gemma4EdgeVisionPooler(config)
        if config.standardize:
            self.register_buffer("std_bias", torch.zeros(config.hidden_size))
            self.register_buffer("std_scale", torch.ones(config.hidden_size))

    def bake_static_mask(
        self, num_patches: int, dtype: torch.dtype = torch.float32
    ) -> None:
        """Convenience wrapper around `self.encoder.bake_static_mask` for
        Stage B's `use_static_mask=True` opt-in. No-op when the flag is
        not set or when `use_fused_sdpa=True`.
        """
        self.encoder.bake_static_mask(num_patches=num_patches, dtype=dtype)

    def bake_sdpa_caches(
        self, num_patches: int, dtype: torch.dtype = torch.float32
    ) -> None:
        """Convenience wrapper around `self.encoder.bake_sdpa_caches` for
        the B5 `use_vulkan_sdpa=True` opt-in. No-op when the flag is off.
        Materializes one (k_cache_scratch, v_cache_scratch) pair per
        attention layer. Step B will share a single pair on the encoder.
        """
        self.encoder.bake_sdpa_caches(num_patches=num_patches, dtype=dtype)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Matches HF Gemma4VisionTower.forward output (last_hidden_state).

        `pixel_values`: `(B, num_patches, 3 * patch_size^2)` post-patchify.
        `pixel_position_ids`: `(B, num_patches, 2)`.
        """
        pooling_k = self.config.pooling_kernel_size
        # HF uses pixel_values.shape[-2] which assumes (B, num_patches, X);
        # output_length follows HF semantics: num_patches divided by k^2.
        # The runtime caller passes pre-flattened pixel_values, so
        # we use shape[1] to be consistent with the HF wrapper at line 1902.
        output_length = pixel_values.shape[-2] // (pooling_k * pooling_k)

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )
        enc_out = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
        )
        hidden_states, pooler_mask = self.pooler(
            hidden_states=enc_out.last_hidden_state,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        hidden_states = hidden_states[pooler_mask]
        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale
        return hidden_states

    # ----- HF -> edge state-dict loader -------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_vision_tower: nn.Module,
        config: Optional[Gemma4EdgeVisionConfig] = None,
    ) -> "Gemma4EdgeVisionTower":
        """Load the edge tower from an HF `Gemma4VisionEncoder` (the class
        the HF main model exposes as `model.vision_tower`).

        - Pulls config fields from `hf_vision_tower.config`.
        - State-dict keys map cleanly: `patch_embedder.*`, `encoder.layers.*`,
          plus `std_bias`/`std_scale` if standardize=True.
        - `Gemma4ClippableLinear` -> `ClippableLinear`: the HF buffers
          `input_min/max`, `output_min/max` are pulled as `.item()` and
          baked into Python scalars in `set_bounds()`.
        """
        hf_cfg = hf_vision_tower.config
        if config is None:
            config = _config_from_hf(hf_cfg)
        edge = cls(config)
        _load_state_from_hf(edge, hf_vision_tower)
        edge.eval()
        return edge


def _config_from_hf(hf_cfg) -> Gemma4EdgeVisionConfig:
    """Build a `Gemma4EdgeVisionConfig` from an HF `Gemma4VisionConfig`."""
    return Gemma4EdgeVisionConfig(
        hidden_size=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_key_value_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.head_dim,
        hidden_activation=hf_cfg.hidden_activation,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        patch_size=hf_cfg.patch_size,
        position_embedding_size=hf_cfg.position_embedding_size,
        rope_theta=float(hf_cfg.rope_parameters["rope_theta"]),
        max_position_embeddings=hf_cfg.max_position_embeddings,
        pooling_kernel_size=hf_cfg.pooling_kernel_size,
        use_clipped_linears=bool(hf_cfg.use_clipped_linears),
        standardize=bool(hf_cfg.standardize),
        attention_dropout=float(getattr(hf_cfg, "attention_dropout", 0.0)),
    )


def _load_state_from_hf(edge: Gemma4EdgeVisionTower, hf_module: nn.Module) -> None:
    """Copy parameters from HF `Gemma4VisionEncoder` into the edge tower.

    Mapping (edge <- HF):
      patch_embedder.input_proj.weight           <- patch_embedder.input_proj.weight
      patch_embedder.position_embedding_table    <- patch_embedder.position_embedding_table
      encoder.layers.{i}.{attn,mlp}.{q|k|v|o|gate|up|down}_proj.linear.weight
                                                <- encoder.layers.{i}.{...}.linear.weight
      encoder.layers.{i}.self_attn.{q,k,v}_norm.weight   <- same
      encoder.layers.{i}.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight   <- same
      std_bias / std_scale                       <- same (if standardize)

    Plus: clipped-linear bound buffers (input_min/max, output_min/max) are
    pulled and baked into the corresponding `ClippableLinear` via
    `set_bounds(...)`.
    """
    hf_sd = hf_module.state_dict()
    edge_sd = edge.state_dict()
    missing_in_hf = []
    unexpected_in_hf = []

    # 1) Plain parameter/weight copy (key names are 1:1 between HF and edge
    #    for the linear weights, layernorms, std_*, position_embedding_table,
    #    input_proj.weight).
    for k in list(edge_sd.keys()):
        if k in hf_sd and hf_sd[k].shape == edge_sd[k].shape:
            edge_sd[k] = hf_sd[k].clone()
        else:
            missing_in_hf.append(k)
    # B1: derive the per-axis split parameters from the loaded table.
    pe_table_key = "patch_embedder.position_embedding_table"
    pe_x_key = "patch_embedder.position_embedding_x"
    pe_y_key = "patch_embedder.position_embedding_y"
    if pe_x_key in edge_sd and pe_table_key in hf_sd:
        edge_sd[pe_x_key] = hf_sd[pe_table_key][0].clone().contiguous()
        edge_sd[pe_y_key] = hf_sd[pe_table_key][1].clone().contiguous()
    edge.load_state_dict(edge_sd, strict=False)

    # 2) Bake ClippableLinear bound buffers from HF.
    # HF parent module path matches: encoder.layers.{i}.{...}.{input,output}_{min,max}
    # but `use_clipped_linears` in HF is registered on Gemma4ClippableLinear,
    # not as a plain weight, so they live in state_dict.
    if not edge.config.use_clipped_linears:
        return

    for edge_name, edge_mod in edge.named_modules():
        if not isinstance(edge_mod, ClippableLinear):
            continue
        # HF path differs only by the absence/presence of `.linear`. The
        # buffers `input_min`, ... in HF live at the parent ClippableLinear
        # path (e.g. encoder.layers.0.self_attn.q_proj.input_min).
        prefix = edge_name  # same path as HF parent
        try:
            imin = float(hf_sd[f"{prefix}.input_min"].item())
            imax = float(hf_sd[f"{prefix}.input_max"].item())
            omin = float(hf_sd[f"{prefix}.output_min"].item())
            omax = float(hf_sd[f"{prefix}.output_max"].item())
        except KeyError:
            # Module without clamping in HF (e.g. patch_embedder.input_proj
            # is a plain nn.Linear). Leave bounds at identity (±inf).
            continue
        edge_mod.set_bounds(imin, imax, omin, omax)


__all__ = [
    "Gemma4EdgeVisionConfig",
    "Gemma4EdgeVisionPatchEmbedder",
    "Gemma4EdgeVisionRotaryEmbedding",
    "Gemma4EdgeVisionAttention",
    "Gemma4EdgeVisionMLP",
    "Gemma4EdgeVisionEncoderLayer",
    "Gemma4EdgeVisionEncoder",
    "Gemma4EdgeVisionPooler",
    "Gemma4EdgeVisionTower",
    "apply_multidim_rope",
    "RMSNorm",
    "ClippableLinear",
]
