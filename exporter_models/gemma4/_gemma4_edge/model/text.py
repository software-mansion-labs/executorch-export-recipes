"""
Gemma 4 — Text Decoder (ExecuTorch edge implementation)
=======================================================

Faithfully matches the reference transformers implementation
(transformers/models/gemma4/modeling_gemma4.py).

Architecture:
  1. Hybrid attention: 4 sliding + 1 global full-context (4:1 pattern)
  2. Dual RoPE: standard θ=10k for sliding, proportional (partial 25%) θ=1M for global
  3. Per-Layer Embeddings (PLE): multiplicative gating with per-layer signal
  4. QKV-Norm: RMSNorm on Q, K (with scale) and V (without scale)
  5. Sandwich norms: pre+post RMSNorm for attention, FFN, and PLE blocks
  6. GQA with different head_dim for sliding (256) vs global (512)
  7. GeGLU FFN with GELU(tanh) activation
  8. Per-layer scalar buffer applied to full layer output
  9. Attention scaling = 1.0 (QKV-norm eliminates need for 1/√d)
  10. Optional logit soft-capping, tied word embeddings
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Gemma4EdgeConfig:
    """Config for Gemma 4 text decoder, matching the official architecture."""

    # --- core transformer dims ---
    vocab_size: int = 262_144
    hidden_size: int = 1536
    num_hidden_layers: int = 35
    num_attention_heads: int = 8       # Q heads
    num_key_value_heads: int = 1       # KV heads (GQA)
    head_dim: int = 256                # sliding attention head dim
    global_head_dim: int = 512         # full attention head dim
    intermediate_size: int = 6144      # FFN hidden (GeGLU)

    # --- PLE (Per-Layer Embeddings) ---
    ple_dim: int = 256                 # hidden_size_per_layer_input
    ple_vocab_size: int = 262_144      # vocab_size_per_layer_input
    use_ple: bool = True

    # --- hybrid attention ---
    sliding_window: int = 512
    max_position_embeddings: int = 131072
    layer_types: Optional[list[str]] = None  # auto: 5:1 pattern

    # --- RoPE ---
    rope_theta: float = 1_000_000.0          # global layers
    rope_local_base_freq: float = 10_000.0   # sliding layers
    partial_rotary_factor: float = 0.25      # fraction of global head_dim rotated

    # --- KV sharing ---
    num_kv_shared_layers: int = 20

    # --- double-wide FFN ---
    use_double_wide_mlp: bool = True

    # --- misc ---
    rms_norm_eps: float = 1e-6
    hidden_activation: str = "gelu_pytorch_tanh"
    final_logit_softcapping: Optional[float] = 30.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.layer_types is None:
            # 4:1 pattern: every 5th layer is full attention (matches real checkpoint)
            sliding_window_pattern = 5
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        # Force last layer to full_attention (matches reference)
        if self.layer_types and self.layer_types[-1] != "full_attention":
            self.layer_types[-1] = "full_attention"

    def ffn_intermediate_size(self, layer_idx: int) -> int:
        """Returns the FFN intermediate size for a given layer."""
        if not self.use_double_wide_mlp:
            return self.intermediate_size
        first_kv_shared = self.num_hidden_layers - self.num_kv_shared_layers
        if self.num_kv_shared_layers > 0 and layer_idx >= first_kv_shared:
            return self.intermediate_size * 2
        return self.intermediate_size


# ---------------------------------------------------------------------------
# RMSNorm — copied from reference Gemma4RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm matching reference (supports with_scale=False for v_norm).

    Two forward paths:
    - default (`use_rsqrt=False`): exact reference `x * pow(variance + eps, -0.5)`.
      `pow(x, -0.5)` drops to portable CPU (XNNPACK only supports integer power==2).
    - rsqrt (`use_rsqrt=True`): mathematically identical `x * rsqrt(variance + eps)`.
      XNNPACK `rsqrt.default` has no shape constraints (3D/4D OK) so the whole
      chain after the mean delegates. `mean(-1)` on 3D still lands in portable
      CPU (XNNPACK `mean.dim` requires 4D output) — accepting that cost keeps
      the change one line with no extra view_copy ops.
    - The failed `use_fused_norm` attempt added 4D reshape to also
      delegate the mean. That produced 2 extra view_copy per RMSNorm × 282
      modules and regressed decode by +16% due to subgraph-boundary overhead,
      so we don't include it.
    """

    use_rsqrt: bool = False  # class-level flag, flipped on all instances
    # Skip the .float() upcast in forward — keep RMSNorm purely in the
    # input dtype. Set True for fp16 Vulkan exports where view_convert
    # shaders are missing. Validated parity-safe at long context (memory
    # note project_gemma4_fp16_residual_safe_long_context).
    no_fp32_upcast: bool = False

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        self.dim = dim
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x*x avoids aten.pow.Tensor_Scalar; bit-equivalent to x.pow(2) for fp.
        var = (x * x).mean(-1, keepdim=True) + self.eps
        if RMSNorm.use_rsqrt:
            return x * torch.rsqrt(var)
        return x * torch.pow(var, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if RMSNorm.no_fp32_upcast:
            output = self._norm(x)
            if self.with_scale:
                output = output * self.weight
            return output
        output = self._norm(x.float())
        if self.with_scale:
            output = output * self.weight.float()
        return output.type_as(x)


class ScaledEmbedding(nn.Embedding):
    """Mirror of HF's Gemma4TextScaledWordEmbedding.

    Why a subclass instead of `embed(ids) * math.sqrt(H)` in forward:
    with a Python-scalar multiply, PyTorch's tensor×scalar semantics
    can keep the scalar at fp32/fp64 precision through the multiply and
    only round the result, so in bf16 weights the effective scale stays
    at the true sqrt(H). HF instead casts the scale tensor to the weight
    dtype first (bf16 rounds sqrt(1536)=39.19→39.25), so the multiply is
    bf16×bf16. The two differ by ~0.15% for hidden_size=1536 and compound
    through every layer. Matching HF here eliminates that drift.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float = 1.0,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        # Stored fp32 to match HF's torch.tensor(scalar) default; cast to
        # weight dtype at multiply-time so rounding matches HF exactly.
        self.register_buffer(
            "embed_scale", torch.tensor(embed_scale, dtype=torch.float32), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings — proportional RoPE with zero-padding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Produces cos/sin tables. Matches reference proportional RoPE:
    rotated dims get real frequencies, non-rotated dims get zeros
    (cos=1, sin=0 → identity).  Denominator uses full dim, not partial.
    """

    def __init__(self, dim: int, base: float, partial_rotary_factor: float = 1.0):
        super().__init__()
        # Number of dims that actually get rotated
        rope_angles = int(partial_rotary_factor * dim // 2)
        # Frequencies — denominator is full dim (matching reference)
        inv_freq_rotated = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2).float() / dim)
        )
        # Zero-pad to full dim//2 so cos/sin match head_dim
        nope_angles = dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat([inv_freq_rotated, torch.zeros(nope_angles)])
        else:
            inv_freq = inv_freq_rotated
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    @torch.no_grad()
    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)              # (seq, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (seq, dim)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,       # (B, H, S, head_dim) or (B, S, H, head_dim)
    cos: torch.Tensor,     # (S, head_dim)
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings. No partial-rotation handling needed:
    zero-padded frequencies produce cos=1, sin=0 → identity.

    Layout is controlled by `Gemma4Attention.use_bshd_kv`:
    - False (default): x is [B, H, S, D]; seq axis=2, head broadcast via unsqueeze(0).
    - True: x is [B, S, H, D]; seq axis=1, head broadcast via unsqueeze(-2).
    """
    if Gemma4Attention.use_prebroadcast_rope:
        # cos/sin already have the 4-D broadcasted shape from the model
        # (see Gemma4EdgeModel.forward / decode_from_embeds). Per-layer
        # unsqueeze + slice are no-ops here — skip both to drop the
        # `unsqueeze_copy.out` (and any residual `slice_copy.Tensor_out`)
        # host ops that FX would otherwise emit per layer.
        pass
    elif Gemma4Attention.use_bshd_kv:
        if Gemma4Attention.skip_redundant_rope_slice:
            cos = cos.unsqueeze(0).unsqueeze(-2)
            sin = sin.unsqueeze(0).unsqueeze(-2)
        else:
            cos = cos[: x.size(1)].unsqueeze(0).unsqueeze(-2)
            sin = sin[: x.size(1)].unsqueeze(0).unsqueeze(-2)
    else:
        if Gemma4Attention.skip_redundant_rope_slice:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos[: x.size(2)].unsqueeze(0).unsqueeze(0)
            sin = sin[: x.size(2)].unsqueeze(0).unsqueeze(0)
    # cos/sin are stored fp32; when x is fp16 the multiply promotes, which
    # breaks downstream torchao int4 linears that require strict-dtype input.
    # Downcast to x.dtype so activations stay in their declared precision.
    return ((x * cos) + (_rotate_half(x) * sin)).to(x.dtype)


# ---------------------------------------------------------------------------
# KV Cache (export-friendly, following ExecuTorch pattern)
# ---------------------------------------------------------------------------

class KVCache(nn.Module):
    """Pre-allocated KV cache with position tracking."""

    def __init__(self, batch_size: int, max_seq_len: int, num_kv_heads: int,
                 head_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.max_seq_len = max_seq_len
        # Storage layout follows Gemma4Attention.use_bshd_kv (class-level flag).
        # - False (BHSD, default): matches Gemma reference shape.
        # - True (BSHD): aligns with custom_sdpa + Llama QuantizedKVCache internal
        #   layout so we skip the ~280 transposes/forward chain.
        if Gemma4Attention.use_bshd_kv:
            cache_shape = (batch_size, max_seq_len, num_kv_heads, head_dim)
        else:
            cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        # NOTE(persistent=True): torch.export tracks mutation through buffers
        # only when they are persistent. With persistent=False, these tensors
        # are lifted as constants; the in-place `index_put_`/`add_` ops in
        # `update()` then mutate read-only constant memory, which the
        # ExecuTorch runtime treats as undefined state on every layer-0 read.
        # That's the root cause of the super-linear eager-vs-PTE drift at
        # increasing depth (uninitialized KV history compounds per layer).
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=True)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=True)
        self.register_buffer("kv_cache_pos", torch.arange(0, max_seq_len), persistent=True)

    def update(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Write at positions given by input_pos, which carries the SYMBOLIC
        # seq-len dim from the exported graph. Previously we sliced
        # self.kv_cache_pos[:seq_len] — kv_cache_pos has concrete shape
        # (max_seq_len,), so during the aten→edge shape_env simplification
        # the symbolic seq-len got substituted with its concrete example
        # value (128), which then specialized every downstream tensor
        # including the aten::embedding.out buffer — making the exported
        # .pte static-only (failed on any seq_len != 128). Using the
        # input_pos tensor directly keeps the dim symbolic.
        if Gemma4Attention.use_bshd_kv:
            self.k_cache[:, input_pos] = k_val
            self.v_cache[:, input_pos] = v_val
        else:
            self.k_cache[:, :, input_pos] = k_val
            self.v_cache[:, :, input_pos] = v_val
        return self.k_cache, self.v_cache

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_cache_pos.copy_(torch.arange(0, self.max_seq_len))


# ---------------------------------------------------------------------------
# Attention — copied from reference Gemma4TextAttention
# ---------------------------------------------------------------------------

class Gemma4Attention(nn.Module):
    """GQA with QKV-norm, hybrid local/global, optional KV cache.

    Key differences from standard attention:
      - scaling = 1.0 (QKV-norm normalizes, no 1/√d needed)
      - v_norm without learnable scale
    """

    # Class-level flag. When True:
    # - KVCache / RingKVCache / QuantizedKVCache* store [B, S, H, D] (BSHD)
    # - apply_rotary_emb reads seq from dim 1 (not dim 2)
    # - Q is kept in [B, S, H, D] (skips the q.transpose(1,2))
    # - K/V skip the pre-cache .transpose(1,2) and the post-cache
    #   .transpose(1,2).contiguous() at the custom_sdpa call site
    # - Non-SDPA fallback (matmul(q, k.T)) gets explicit transposes at the
    #   matmul site to keep BHSD semantics there.
    # Llama's QuantizedKVCache already stores BSHD internally and pays two
    # transposes per update/return. Turning this flag on removes those too.
    use_bshd_kv: bool = False

    # Class-level flag. When True, global (full-max_seq_len cache) layers
    # slice K/V and the static mask to `effective_len = input_pos[-1]+1`
    # before attention. At decode pos P this cuts per-step global-attention
    # work from O(max_seq_len) to O(P), which is a 4× win at P=max_seq_len/4
    # and grows with the max_seq_len : current_pos ratio. Ring (sliding)
    # layers already attend to exactly 2W slots so they're not sliced.
    use_dynamic_kv_slice: bool = False

    # Class-level flag. When True, the static `[max_seq_len, max_seq_len]`
    # attention mask is stored as bool (True == masked) instead of an
    # additive fp32 tensor with 0/-inf values. Cuts that buffer 32×
    # (fp32=4B/cell → bool=1B/cell). At use, non-SDPA path uses masked_fill;
    # custom_sdpa path materializes a fp mask from the sliced row.
    use_bool_mask: bool = False

    # Class-level flag. When True, global (full_attention) layers skip the
    # static `[max_seq_len, max_seq_len]` mask buffer entirely — forward
    # rebuilds the `[S, max_seq_len]` rows on-the-fly via arange+compare.
    # Eliminates the PTE-constant region reserved for the causal mask
    # (after FlatBuffer dedup the seven global layers share one buffer).
    # Orthogonal to use_bool_mask: that flag still selects bool vs additive
    # at use-site. Sliding layers are unaffected — they rely on
    # replace_sliding_kv_cache_with_ring_kv_cache for JIT mask generation.
    use_jit_mask: bool = False

    # Class-level flag. When True, the ring-cache index + mask computation
    # is hoisted to model-level (Gemma4EdgeModel.forward /
    # decode_from_embeds) and the result is broadcast to every sliding
    # (ring) layer via new `precomputed_indices` / `precomputed_mask`
    # kwargs on the cache update / make_attention_mask paths.
    #
    # All ring layers share identical `cache_positions` state (init -1,
    # reset -1, only writer is index_copy_(0, indices, orig_indices) with
    # identical args), so the per-layer recomputation is redundant. Hoist
    # cuts (N_ring - 1) duplicate calls per forward.
    use_shared_ring_state: bool = False

    # Class-level flag. When True, `apply_rotary_emb` drops the
    # `cos[: x.size(N)]` / `sin[: x.size(N)]` slice. In every export-path
    # caller (forward, decode_from_embeds, gemma_sdpa_source_transform),
    # cos/sin are constructed at exactly the input's seq_len (via
    # `F.embedding(input_pos, ...)`, `self._cos_*[input_pos]`, or
    # `self.rope_*(S, device)`), so the slice is functionally a no-op
    # but FX still emits a `slice_copy.Tensor_out` per call. With
    # use_bshd_kv=True at 5L this removes 20 host slice ops (10 cos +
    # 10 sin, q+k per layer). The flag exists for safety: any caller
    # that passes a cos/sin table larger than `x.size(seq_axis)` must
    # leave it OFF. No such caller exists in the repo.
    skip_redundant_rope_slice: bool = False

    # Class-level flag. When True, the broadcasting unsqueezes in
    # `apply_rotary_emb` (`cos.unsqueeze(0).unsqueeze(-2|0)` and the
    # matching sin) are hoisted to model level. `Gemma4EdgeModel.forward`
    # and `decode_from_embeds` precompute the broadcasted-shape cos/sin
    # once per forward and pass them via the existing `cos` / `sin`
    # kwargs — no new kwargs needed. `apply_rotary_emb` then sees a
    # cos/sin already in 4-D and skips the unsqueeze block.
    #
    # Each layer otherwise applies 4 `unsqueeze_copy` host ops (2× cos +
    # 2× sin) per Q AND K path. With use_bshd_kv=True + the source
    # transform's fused SDPA path the count collapses to 4 unsqueezes
    # per non-shared layer × 2 (Q+K = same cos/sin, the unsqueezes are
    # graph-shared inside one apply_rotary_emb call). At 5L a profile
    # counted 41 `aten::unsqueeze_copy.out`. Hoisting cuts this to a
    # constant 4 (one cos_local + sin_local + cos_global + sin_global
    # broadcasted set, shared by every layer). Composes with Lever E
    # (skip_redundant_rope_slice).
    use_prebroadcast_rope: bool = False

    # NOTE: `use_rotate_half_matmul` (Lever G) used to live here. It rewrote
    # `_rotate_half` as a matmul against a `[D, D]` permutation basis. It was
    # FALSIFIED by measurement and is deliberately not carried into this
    # package:
    # eager was bit-identical, but the PTE regressed (host ops 411 -> 422,
    # arena 90.98 -> 94.27 MB) because `neg`/`cat` were already absorbed into
    # XNNPACK delegates and the 4D->3D->2D `et_view` reshape tax at each
    # matmul boundary (+20) swamped the 10 ops it removed. See docs/LEVERS.md.

    def __init__(self, config: Gemma4EdgeConfig, layer_idx: int, is_global: bool):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_global = is_global
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.kv_group_size = self.num_heads // self.num_kv_heads

        # Head dim depends on layer type (reference line 1137)
        self.head_dim = config.global_head_dim if is_global else config.head_dim

        # KV-sharing flags (reference lines 1148-1160).
        # Layers from [num_hidden_layers - num_kv_shared_layers, num_hidden_layers)
        # reuse K/V from the last non-shared layer of the same layer_type.
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = (
            config.num_kv_shared_layers > 0 and layer_idx >= first_shared
        )
        prev_types = config.layer_types[:first_shared]
        ltype = config.layer_types[layer_idx]
        if self.is_kv_shared_layer:
            self.kv_shared_layer_index = len(prev_types) - 1 - prev_types[::-1].index(ltype)
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            self.store_full_length_kv = (
                ltype in prev_types
                and layer_idx == len(prev_types) - 1 - prev_types[::-1].index(ltype)
            )

        # Projections (reference lines 1166-1177)
        # Weights are kept on shared layers too (real checkpoint contains them);
        # HF loads but never calls them on shared layers.
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # QKV-Norm (reference lines 1162-1164)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

        # Attention scaling = 1.0 (reference line 1143)
        self.scale = 1.0

        # When True, skip explicit K/V head replication (repeat_interleave)
        # and rely on torch.matmul's batch-dim broadcasting. Profiling put the
        # explicit materialization at ~17% of portable-CPU time in prefill.
        self.use_broadcast_gqa: bool = False

        # When True, replace the manual Q·K^T + mask + softmax + @V chain with
        # executorch's `torch.ops.llama.custom_sdpa` fused op. Requires
        # _use_cache=True (the op consumes a pre-concatenated K/V tensor and a
        # 2D additive mask). Caller must import `executorch.extension.llm.custom_ops`
        # before torch.export so the op is registered.
        self.use_custom_sdpa: bool = False

        # KV cache — not allocated for shared layers (they read from source).
        self.kv_cache: Optional[KVCache] = None
        self._use_cache = False
        # `kv_source` is wired by Gemma4EdgeModel.__init__ and stored via
        # object.__setattr__ to bypass nn.Module's submodule auto-registration.
        # Registering it as a submodule would duplicate every q/k/v/o_proj
        # weight in state_dict under `…kv_source.…`, breaking
        # load_state_dict(strict=False) with ~120 spurious missing keys.
        # torch.export still traces reads via self.kv_source._last_k / .kv_cache
        # because the access is resolved at trace time into concrete tensors.
        object.__setattr__(self, "kv_source", None)

    def setup_caches(self, batch_size: int, max_seq_len: int, dtype: torch.dtype) -> None:
        self._use_cache = True
        self._max_seq_len = max_seq_len
        self._mask_dtype = dtype
        # Every layer gets its own KVCache, including KV-shared
        # consumers, to avoid the Vulkan partitioner duplicating a shared
        # cache placeholder into per-partition unwritten copies. Consumer
        # layers write shared_k/shared_v (from source) into their own
        # cache via gemma_sdpa_with_kv_cache, then read from it. Same K/V
        # values, distinct buffers — correctness preserved at ~2× cache
        # memory cost.
        self.kv_cache = KVCache(batch_size, max_seq_len, self.num_kv_heads, self.head_dim, dtype)
        if Gemma4Attention.use_jit_mask and self.is_global:
            # Skip the static buffer — `_jit_causal_mask_rows` regenerates
            # the [S, max_seq_len] rows per forward. Keep `self.mask = None`
            # so attribute access stays valid; the forward branches check
            # `use_jit_mask and self.is_global` before touching it.
            self.mask = None
            return
        if Gemma4Attention.use_bool_mask:
            if self.is_global:
                mask = torch.triu(
                    torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
                    diagonal=1,
                )
            else:
                window = self.config.sliding_window
                row = torch.arange(max_seq_len).unsqueeze(1)
                col = torch.arange(max_seq_len).unsqueeze(0)
                mask = (col > row) | (row - col >= window)
            self.register_buffer("mask", mask, persistent=False)
        else:
            if self.is_global:
                mask = torch.triu(
                    torch.full((max_seq_len, max_seq_len), float("-inf")),
                    diagonal=1,
                )
            else:
                window = self.config.sliding_window
                row = torch.arange(max_seq_len).unsqueeze(1)
                col = torch.arange(max_seq_len).unsqueeze(0)
                mask = torch.where(
                    (col > row) | (row - col >= window),
                    torch.tensor(float("-inf")),
                    torch.tensor(0.0),
                )
            self.register_buffer("mask", mask.to(dtype), persistent=False)

    def _jit_causal_mask_rows(self, input_pos: torch.Tensor) -> torch.Tensor:
        """Rebuild the `[S, max_seq_len]` slice that `self.mask[input_pos]`
        would produce for a global (full-attention) layer, without storing
        the static buffer. `input_pos` is shape `(S,)`. Returns a bool tensor
        when `use_bool_mask` is set, otherwise an additive fp tensor with
        `-inf` at masked positions."""
        max_len = self._max_seq_len
        col = torch.arange(max_len, device=input_pos.device)
        row = input_pos.unsqueeze(-1)
        bool_mask = col.unsqueeze(0) > row
        if Gemma4Attention.use_bool_mask:
            return bool_mask
        return torch.where(
            bool_mask,
            torch.tensor(float("-inf"), dtype=self._mask_dtype),
            torch.tensor(0.0, dtype=self._mask_dtype),
        )

    def forward(
        self,
        hidden: torch.Tensor,             # (B, S, hidden_size)
        cos: torch.Tensor,
        sin: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        shared_k: Optional[torch.Tensor] = None,
        shared_v: Optional[torch.Tensor] = None,
        ring_indices: Optional[torch.Tensor] = None,
        ring_mask: Optional[torch.Tensor] = None,
    ):
        # Returns (out, k_export, v_export). `k_export`/`v_export` are the
        # post-RoPE, post-kv-cache-update K/V tensors — already
        # attention-ready (broadcast shape (B, num_kv_heads, T_full, head_dim)).
        # The caller (Gemma4EdgeModel.forward) threads them into later shared
        # layers via the `shared_k/shared_v` kwargs. Shared layers MUST be
        # given non-None tensors; non-shared layers receive None.
        #
        # This replaces the older `self._last_k`/`self.kv_source._last_v`
        # attribute-stash pattern, which torch.export silently dropped:
        # during strict=True tracing, attribute assignments on `self`
        # become warnings ("must be registered as buffers") and the reads
        # in shared layers captured stale zeros. Threading K/V as SSA
        # values makes the dataflow explicit in the exported graph.
        B, S, _ = hidden.shape

        # --- Q: always computed per-layer (reference lines 1192-1195) ---
        q = self.q_proj(hidden).view(B, S, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        if not Gemma4Attention.use_bshd_kv:
            q = q.transpose(1, 2)
        q = apply_rotary_emb(q, cos, sin)

        # --- K/V: shared layers reuse the upstream source's K/V; all
        #         others compute their own (reference lines 1198-1202). ---
        if self.is_kv_shared_layer:
            k = shared_k
            v = shared_v
        else:
            k = self.k_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
            v = self.v_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
            k = self.k_norm(k)
            v = self.v_norm(v)
            if not Gemma4Attention.use_bshd_kv:
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
            k = apply_rotary_emb(k, cos, sin)

            if self._use_cache:
                assert input_pos is not None, "KV cache update requires input_pos"
                # Shared-state path: when ring_indices is provided, the cache
                # consumes precomputed indices and skips its own bookkeeping.
                if ring_indices is not None and getattr(
                    self.kv_cache, "is_ring_buffer", False
                ):
                    k, v = self.kv_cache.update(
                        k, v, input_pos, precomputed_indices=ring_indices,
                    )
                else:
                    k, v = self.kv_cache.update(k, v, input_pos)

        # --- GQA head repeat ---
        # When use_broadcast_gqa is True, skip materialization: torch.matmul
        # broadcasts the kv-heads dim [B, 1, …] against q's [B, num_heads, …].
        # custom_sdpa also broadcasts GQA internally, so it doesn't need k_rep/v_rep.
        if self.use_custom_sdpa and self._use_cache:
            # --- Fused SDPA path ---
            # Build the 2D additive mask [S, L] that custom_sdpa expects.
            src_cache = self.kv_source.kv_cache if self.is_kv_shared_layer else self.kv_cache
            is_ring = getattr(src_cache, "is_ring_buffer", False)
            if is_ring:
                # Shared-state path: hand the precomputed mask straight through
                # (cache.make_attention_mask just returns it). When ring_mask
                # is None, cache rebuilds it the original way.
                mask_2d = src_cache.make_attention_mask(
                    input_pos, precomputed_mask=ring_mask,
                )
            elif Gemma4Attention.use_jit_mask and self.is_global:
                mask_2d = self._jit_causal_mask_rows(input_pos)
            else:
                mask_2d = self.mask[input_pos]
            # custom_sdpa expects [B, S, H, D]. With use_bshd_kv=True the
            # tensors are already BSHD (and contiguous out of the cache), so
            # we feed them directly and save the transpose+contiguous copy —
            # at decode this is ~70 MB/k and ~70 MB/v full-cache traffic.
            if Gemma4Attention.use_bshd_kv:
                q_bshd = q
                k_bshd = k
                v_bshd = v
            else:
                q_bshd = q.transpose(1, 2).contiguous()
                k_bshd = k.transpose(1, 2).contiguous()
                v_bshd = v.transpose(1, 2).contiguous()
            # Dynamic KV slicing: for global layers with a static mask, narrow
            # K/V and the mask to [:, :effective_len] so attention scans only
            # the populated prefix of the cache. Ring layers already operate
            # on the tight 2W window, so skip them. Only applies when cache
            # is being used (prefill-without-cache path falls through).
            if (Gemma4Attention.use_dynamic_kv_slice
                    and self.is_global and not is_ring):
                effective_len = int(input_pos[-1].item()) + 1
                torch._check_is_size(effective_len)
                # slice_forward decomp guards `end <= max_dim`; teach the
                # shape env that effective_len is within the cache bound.
                torch._check(effective_len <= k_bshd.shape[1])
                torch._check(effective_len <= mask_2d.shape[1])
                # Stack K/V and slice once so both share a single symbolic
                # size — two independent slices produce distinct unbacked
                # SymInts, which breaks custom_sdpa._validate_params'
                # `key_cache.size() == value_cache.size()` tuple compare.
                kv = torch.stack([k_bshd, v_bshd], dim=0)
                kv = kv[:, :, :effective_len]
                k_bshd = kv[0]
                v_bshd = kv[1]
                mask_2d = mask_2d[:, :effective_len]
            if Gemma4Attention.use_bool_mask and not is_ring:
                # custom_sdpa wants an additive fp mask; materialize after
                # slicing so the conversion only runs on the live window.
                mask_2d = torch.zeros_like(
                    mask_2d, dtype=q_bshd.dtype,
                ).masked_fill(mask_2d, float("-inf"))
            # custom_sdpa's C++ backend + Python meta kernel both require
            # fp32 q/k/v. For fp16 callers upcast at the boundary and
            # downcast the output so residual/KV stay in fp16 but sdpa
            # still executes in the only dtype it supports.
            caller_dtype = q_bshd.dtype
            needs_upcast = caller_dtype != torch.float32
            if needs_upcast:
                q_sdpa = q_bshd.float()
                # Cast k_bshd/v_bshd ONLY if they're not already fp32. With an
                # fp32 KV cache (as an SDPA-fusion recipe produces), `k_bshd`
                # comes from `kv_cache.update` returning self.k_cache which
                # is fp32; an unconditional .float() would emit a no-op
                # `_to_copy(self.k_cache)` during torch.export, breaking the
                # Vulkan CausalSDPAMatch pattern (which requires
                # custom_sdpa.args[1] to be the SAME fx.Node that
                # update_cache mutates). Per-tensor dtype check keeps the
                # cast on fp16-cached paths and skips it on fp32-cached
                # paths.
                k_sdpa = k_bshd.float() if k_bshd.dtype != torch.float32 else k_bshd
                v_sdpa = v_bshd.float() if v_bshd.dtype != torch.float32 else v_bshd
                mask_sdpa = mask_2d.float() if mask_2d is not None else None
            else:
                q_sdpa, k_sdpa, v_sdpa, mask_sdpa = q_bshd, k_bshd, v_bshd, mask_2d
            out_bshd = torch.ops.llama.custom_sdpa(
                q_sdpa, k_sdpa, v_sdpa,
                0,
                mask_sdpa,
                0.0,
                False,
                float(self.scale),
            )
            if needs_upcast:
                out_bshd = out_bshd.to(caller_dtype)
            out = out_bshd.reshape(B, S, -1)
            return self.o_proj(out), k, v

        # Non-SDPA fallback assumes BHSD for the Q·K^T matmul and the head
        # dim == dim 1 for repeat_interleave. If the caller kept Q/K/V in
        # BSHD (use_bshd_kv=True), transpose locally — the returned
        # (k, v) at the end of forward keep the caller's layout.
        if Gemma4Attention.use_bshd_kv:
            q_attn = q.transpose(1, 2)
            k_attn = k.transpose(1, 2)
            v_attn = v.transpose(1, 2)
        else:
            q_attn = q
            k_attn = k
            v_attn = v

        # Dynamic KV slicing for the non-SDPA fallback. Same conditions as
        # the SDPA branch: only for global, non-ring, cached layers.
        src_cache = self.kv_source.kv_cache if self.is_kv_shared_layer else self.kv_cache
        is_ring = getattr(src_cache, "is_ring_buffer", False)
        slice_kv = (
            self._use_cache
            and Gemma4Attention.use_dynamic_kv_slice
            and self.is_global
            and not is_ring
        )
        if slice_kv:
            effective_len = int(input_pos[-1].item()) + 1
            torch._check_is_size(effective_len)
            # BHSD after the transpose block above: seq axis = dim 2.
            k_attn = k_attn[:, :, :effective_len]
            v_attn = v_attn[:, :, :effective_len]

        if self.kv_group_size > 1 and not self.use_broadcast_gqa:
            k_rep = k_attn.repeat_interleave(self.kv_group_size, dim=1)
            v_rep = v_attn.repeat_interleave(self.kv_group_size, dim=1)
        else:
            k_rep = k_attn
            v_rep = v_attn

        # --- Attention + mask ---
        attn_weights = torch.matmul(q_attn, k_rep.transpose(-2, -1)) * self.scale
        if self._use_cache:
            # Ring-buffer caches compute the mask dynamically because the
            # ring slot ↔ absolute-position mapping changes each step. The
            # default static mask buffer is sized [max_seq_len, max_seq_len]
            # and is dropped at swap time, so we must NOT touch self.mask
            # in the ring branch. Shared layers never own a ring cache, so
            # they fall through to the static `self.mask` path.
            if is_ring:
                ring_mask = src_cache.make_attention_mask(input_pos)
                attn_weights = attn_weights + ring_mask.unsqueeze(0).unsqueeze(0)
            else:
                if Gemma4Attention.use_jit_mask and self.is_global:
                    mask_row = self._jit_causal_mask_rows(input_pos)
                else:
                    mask_row = self.mask[input_pos]
                if slice_kv:
                    mask_row = mask_row[:, :effective_len]
                if Gemma4Attention.use_bool_mask:
                    attn_weights = attn_weights.masked_fill(
                        mask_row.unsqueeze(0).unsqueeze(0), float("-inf"),
                    )
                else:
                    attn_weights = attn_weights + mask_row.unsqueeze(0).unsqueeze(0)
        else:
            if not self.is_global:
                window = self.config.sliding_window
                row_idx = torch.arange(S, device=hidden.device).unsqueeze(1)
                col_idx = torch.arange(S, device=hidden.device).unsqueeze(0)
                sw_mask = (col_idx > row_idx) | (row_idx - col_idx >= window)
                attn_weights = attn_weights.masked_fill(sw_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            else:
                causal = torch.triu(torch.ones(S, S, device=hidden.device, dtype=torch.bool), diagonal=1)
                attn_weights = attn_weights.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Reference (modeling_gemma4.py:795) upcasts softmax to fp32 and
        # casts back. Critical for quantized-weight runs where accumulated
        # pre-softmax errors compress into 0/1 saturation in fp16.
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_weights, v_rep)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out), k, v


# ---------------------------------------------------------------------------
# Feed-Forward Network (GeGLU) — copied from reference Gemma4TextMLP
# ---------------------------------------------------------------------------

class Gemma4FFN(nn.Module):
    """GeGLU FFN: out = down(GELU(gate(x)) * up(x))"""

    def __init__(self, config: Gemma4EdgeConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder Layer — copied from reference Gemma4TextDecoderLayer
# ---------------------------------------------------------------------------

class Gemma4DecoderLayer(nn.Module):
    """Decoder layer: attention → FFN → PLE → layer_scalar.

    Matches reference lines 1319-1403:
      - PLE is AFTER FFN (multiplicative gating)
      - layer_scalar is a buffer applied to full layer output
    """

    def __init__(self, config: Gemma4EdgeConfig, layer_idx: int, is_global: bool):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = Gemma4Attention(config, layer_idx, is_global=is_global)

        # FFN
        self.mlp = Gemma4FFN(config, intermediate_size=config.ffn_intermediate_size(layer_idx))

        # Sandwich norms (reference lines 1327-1330)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Layer scalar — buffer, applied to full output (reference line 1331, 1402)
        self.register_buffer("layer_scalar", torch.ones(1))

        # PLE per-layer components (reference lines 1334-1338)
        self.use_ple = config.use_ple
        if config.use_ple:
            self.per_layer_input_gate = nn.Linear(config.hidden_size, config.ple_dim, bias=False)
            self.per_layer_projection = nn.Linear(config.ple_dim, config.hidden_size, bias=False)
            self.post_per_layer_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        ple_signal: Optional[torch.Tensor] = None,
        shared_k: Optional[torch.Tensor] = None,
        shared_v: Optional[torch.Tensor] = None,
        ring_indices: Optional[torch.Tensor] = None,
        ring_mask: Optional[torch.Tensor] = None,
    ):
        # Returns (hidden, k_export, v_export) — k/v flow to the model's
        # kv-by-type tracker for downstream shared layers.
        # --- Attention block (sandwich norm, reference lines 1358-1370) ---
        residual = hidden
        hidden = self.input_layernorm(hidden)
        attn_out, k_out, v_out = self.self_attn(
            hidden, cos, sin, input_pos, shared_k, shared_v,
            ring_indices=ring_indices, ring_mask=ring_mask,
        )
        hidden = self.post_attention_layernorm(attn_out)
        hidden = residual + hidden

        # --- FFN block (sandwich norm, reference lines 1372-1391) ---
        residual = hidden
        hidden = self.pre_feedforward_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = self.post_feedforward_layernorm(hidden)
        hidden = residual + hidden

        # --- PLE injection AFTER FFN (reference lines 1393-1400) ---
        if self.use_ple and ple_signal is not None:
            residual = hidden
            hidden = self.per_layer_input_gate(hidden)          # (B,S,hidden) → (B,S,ple_dim)
            hidden = F.gelu(hidden, approximate="tanh")         # activation
            hidden = hidden * ple_signal                         # MULTIPLY by PLE signal
            hidden = self.per_layer_projection(hidden)          # (B,S,ple_dim) → (B,S,hidden)
            hidden = self.post_per_layer_input_norm(hidden)
            hidden = residual + hidden

        # Layer scalar on full output (reference line 1402)
        hidden = hidden * self.layer_scalar

        return hidden, k_out, v_out


# ---------------------------------------------------------------------------
# Full Text Decoder Model — copied from reference Gemma4TextModel
# ---------------------------------------------------------------------------

class Gemma4EdgeModel(nn.Module):
    """Gemma 4 text decoder. PLE signal computation matches reference
    get_per_layer_inputs() + project_per_layer_inputs()."""

    def __init__(self, config: Gemma4EdgeConfig):
        super().__init__()
        self.config = config

        # Token embedding (reference lines 1505-1507). ScaledEmbedding applies
        # `* sqrt(hidden_size)` *inside* the module with the scale cast to
        # weight.dtype — matching HF's rounding semantics exactly.
        self.embed_tokens = ScaledEmbedding(
            config.vocab_size, config.hidden_size,
            embed_scale=math.sqrt(config.hidden_size),
        )

        # PLE (reference lines 1518-1531)
        if config.use_ple:
            total_ple_dim = config.num_hidden_layers * config.ple_dim
            self.embed_tokens_per_layer = ScaledEmbedding(
                config.ple_vocab_size, total_ple_dim,
                embed_scale=math.sqrt(config.ple_dim),
            )
            self.per_layer_model_projection = nn.Linear(config.hidden_size, total_ple_dim, bias=False)
            self.per_layer_projection_norm = RMSNorm(config.ple_dim, eps=config.rms_norm_eps)
            self._ple_split = False

        # Layer schedule
        self.layer_is_global: list[bool] = [
            t == "full_attention" for t in config.layer_types
        ]

        # Decoder layers
        self.layers = nn.ModuleList([
            Gemma4DecoderLayer(config, i, self.layer_is_global[i])
            for i in range(config.num_hidden_layers)
        ])

        # Wire KV-sharing source references (reference lines 1198-1202).
        # Shared layers pull K/V from the last non-shared layer of the same
        # type; the module-reference wiring gives torch.export a static
        # tensor-attribute access path rather than a dict threaded through
        # forward signatures.
        if config.num_kv_shared_layers > 0:
            for i in range(config.num_hidden_layers):
                attn = self.layers[i].self_attn
                if attn.is_kv_shared_layer:
                    object.__setattr__(
                        attn,
                        "kv_source",
                        self.layers[attn.kv_shared_layer_index].self_attn,
                    )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Dual RoPE — proportional for global, default for sliding
        self.rope_local = RotaryEmbedding(
            config.head_dim, base=config.rope_local_base_freq, partial_rotary_factor=1.0,
        )
        self.rope_global = RotaryEmbedding(
            config.global_head_dim, base=config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
        )

        self._use_kv_cache = False

    def setup_caches(self, batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.float32) -> None:
        for layer in self.layers:
            layer.self_attn.setup_caches(batch_size, max_seq_len, dtype)
        self._use_kv_cache = True

        device = self.embed_tokens.weight.device
        cos_local, sin_local = self.rope_local(max_seq_len, device)
        cos_global, sin_global = self.rope_global(max_seq_len, device)
        self.register_buffer("_cos_local", cos_local, persistent=False)
        self.register_buffer("_sin_local", sin_local, persistent=False)
        self.register_buffer("_cos_global", cos_global, persistent=False)
        self.register_buffer("_sin_global", sin_global, persistent=False)

        # Shared ring state buffers — read/written only when
        # `Gemma4Attention.use_shared_ring_state` is True (see decode_from_embeds /
        # forward). When the flag is off these buffers are dead weight; when on
        # they replace the per-layer cache_positions + kv_cache_pos updates.
        # All ring layers in Gemma4 share the same sliding_window so a single
        # pair suffices. Sliding-only — global layers never read these.
        if any(not g for g in self.layer_is_global):
            max_context_length = 2 * self.config.sliding_window
            self.register_buffer(
                "_shared_ring_cache_positions",
                torch.full((max_context_length,), -1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_shared_ring_kv_cache_pos",
                torch.arange(0, max_context_length),
                persistent=False,
            )
            self._shared_ring_window = self.config.sliding_window
            self._shared_ring_max_context_length = max_context_length
            self._shared_ring_dtype = dtype
        else:
            self._shared_ring_window = None
            self._shared_ring_max_context_length = None
            self._shared_ring_dtype = None

    def reset_caches(self) -> None:
        for layer in self.layers:
            if layer.self_attn.kv_cache is not None:
                layer.self_attn.kv_cache.reset()
        if getattr(self, "_shared_ring_max_context_length", None) is not None:
            self._shared_ring_cache_positions.fill_(-1)
            self._shared_ring_kv_cache_pos.copy_(
                torch.arange(0, self._shared_ring_max_context_length)
            )

    def split_ple_for_export(self) -> None:
        """Split PLE weights into per-layer nn.Embedding / nn.Linear modules.

        torch.export lifts parameters into the graph as constants. Slicing
        a lifted constant at runtime (narrow / getitem) is not supported by
        the edge dialect. Pre-splitting gives each layer its own constant.

        Must be called BEFORE quantization so torchao quantizes each per-layer
        module independently. The names contain "per_layer" so the existing
        quantization filters still match.
        """
        if not self.config.use_ple or self._ple_split:
            return

        d = self.config.ple_dim
        L = self.config.num_hidden_layers
        V = self.config.ple_vocab_size
        H = self.config.hidden_size

        emb_data = self.embed_tokens_per_layer.weight.data
        proj_data = self.per_layer_model_projection.weight.data

        # [vocab, L*d] → L × ScaledEmbedding(vocab, d)
        emb_chunks = emb_data.split(d, dim=1)
        self._ple_per_layer_embs = nn.ModuleList()
        for chunk in emb_chunks:
            emb = ScaledEmbedding(V, d, embed_scale=math.sqrt(d))
            emb.weight = nn.Parameter(chunk.contiguous())
            self._ple_per_layer_embs.append(emb)

        # [L*d, hidden] → L × nn.Linear(hidden, d, bias=False)
        proj_chunks = proj_data.split(d, dim=0)
        self._ple_per_layer_projs = nn.ModuleList()
        for chunk in proj_chunks:
            proj = nn.Linear(H, d, bias=False)
            proj.weight = nn.Parameter(chunk.contiguous())
            self._ple_per_layer_projs.append(proj)

        # Remove originals so they don't duplicate in the exported graph
        del self.embed_tokens_per_layer
        del self.per_layer_model_projection
        self._ple_split = True

    def forward(
        self,
        input_ids: torch.Tensor,                       # (B, S)
        input_pos: Optional[torch.Tensor] = None,      # (S,)
    ) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device

        # --- Embeddings (scaled by sqrt(hidden_size) inside ScaledEmbedding,
        #     matching HF's Gemma4TextScaledWordEmbedding semantics). ---
        hidden = self.embed_tokens(input_ids)
        # PLE projections must be computed from the *initial* embedded tokens,
        # not from the evolving hidden state inside the layer loop. Cache once.
        hidden_for_ple = hidden

        # --- RoPE cos/sin ---
        if self._use_kv_cache:
            # F.embedding lowers to aten.embedding.default (Vulkan-supported);
            # `tensor[input_pos]` lowers to aten.index.Tensor which Vulkan rejects.
            cos_local = F.embedding(input_pos, self._cos_local)
            sin_local = F.embedding(input_pos, self._sin_local)
            cos_global = F.embedding(input_pos, self._cos_global)
            sin_global = F.embedding(input_pos, self._sin_global)
        else:
            cos_local, sin_local = self.rope_local(S, device)
            cos_global, sin_global = self.rope_global(S, device)

        # Lever F: pre-broadcast cos/sin to the 4-D shape that
        # apply_rotary_emb would otherwise unsqueeze per-layer. Done once
        # per forward so every layer (and Q+K paths) consume the same
        # broadcasted tensors instead of re-unsqueezing 4× per layer.
        if Gemma4Attention.use_prebroadcast_rope:
            if Gemma4Attention.use_bshd_kv:
                # x is [B, S, H, D] → seq axis 1; head broadcast dim -2.
                # cos/sin: [S, D] → [1, S, 1, D].
                cos_local = cos_local.unsqueeze(0).unsqueeze(-2)
                sin_local = sin_local.unsqueeze(0).unsqueeze(-2)
                cos_global = cos_global.unsqueeze(0).unsqueeze(-2)
                sin_global = sin_global.unsqueeze(0).unsqueeze(-2)
            else:
                # x is [B, H, S, D] → seq axis 2; head broadcast dim 0/1.
                # cos/sin: [S, D] → [1, 1, S, D].
                cos_local = cos_local.unsqueeze(0).unsqueeze(0)
                sin_local = sin_local.unsqueeze(0).unsqueeze(0)
                cos_global = cos_global.unsqueeze(0).unsqueeze(0)
                sin_global = sin_global.unsqueeze(0).unsqueeze(0)

        # --- PLE: all-at-once when quantized, per-layer when split ---
        per_layer_inputs = None
        d = self.config.ple_dim
        if self.config.use_ple and not self._ple_split:
            # Quantized PLE: full embedding lookup (torchao ops trace correctly).
            # Scale by sqrt(ple_dim) is applied inside ScaledEmbedding.
            ple_tok = self.embed_tokens_per_layer(input_ids)  # (B, S, L*d)
            ple_tok = ple_tok.reshape(B, S, self.config.num_hidden_layers, d)

            ple_ctx = self.per_layer_model_projection(hidden)  # (B, S, L*d)
            ple_ctx = ple_ctx * (self.config.hidden_size ** -0.5)
            ple_ctx = ple_ctx.reshape(B, S, self.config.num_hidden_layers, d)
            ple_ctx = self.per_layer_projection_norm(ple_ctx)

            per_layer_inputs = (ple_ctx + ple_tok) * (2.0 ** -0.5)  # (B, S, L, d)

        # --- Forward through layers ---
        # Track per-type K/V from the most recent source layer so shared
        # layers can read them via explicit dataflow (replaces the older
        # self._last_k attribute-stash pattern that torch.export dropped).
        sliding_k: Optional[torch.Tensor] = None
        sliding_v: Optional[torch.Tensor] = None
        full_k: Optional[torch.Tensor] = None
        full_v: Optional[torch.Tensor] = None

        # Lever B+C: compute ring indices + mask once per forward (only when
        # ring caches present + flag enabled). Mirrors decode_from_embeds.
        ring_indices: Optional[torch.Tensor] = None
        ring_mask: Optional[torch.Tensor] = None
        if (
            self._use_kv_cache
            and Gemma4Attention.use_shared_ring_state
            and getattr(self, "_shared_ring_max_context_length", None) is not None
            and input_pos is not None
        ):
            from .transforms.kv_ring import compute_shared_ring_state
            ring_indices, ring_mask = compute_shared_ring_state(
                self._shared_ring_cache_positions,
                self._shared_ring_kv_cache_pos,
                input_pos,
                self._shared_ring_window,
                self._shared_ring_max_context_length,
                self._shared_ring_dtype,
            )

        for i, layer in enumerate(self.layers):
            is_global = self.layer_is_global[i]
            cos = cos_global if is_global else cos_local
            sin = sin_global if is_global else sin_local

            ple_signal = None
            if self.config.use_ple:
                if self._ple_split:
                    # Per-layer modules: activation is (B, S, d) not (B, S, L*d).
                    # Scale by sqrt(ple_dim) is applied inside ScaledEmbedding.
                    ple_tok_i = self._ple_per_layer_embs[i](input_ids)

                    ple_ctx_i = self._ple_per_layer_projs[i](hidden_for_ple)
                    ple_ctx_i = ple_ctx_i * (self.config.hidden_size ** -0.5)
                    ple_ctx_i = self.per_layer_projection_norm(ple_ctx_i)

                    ple_signal = (ple_ctx_i + ple_tok_i) * (2.0 ** -0.5)
                else:
                    # Fallback: slice from precomputed (B, S, L, d)
                    ple_signal = per_layer_inputs[:, :, i, :]

            attn = layer.self_attn
            if attn.is_kv_shared_layer:
                if is_global:
                    shared_k, shared_v = full_k, full_v
                else:
                    shared_k, shared_v = sliding_k, sliding_v
            else:
                shared_k, shared_v = None, None

            layer_ring_indices = None if is_global else ring_indices
            layer_ring_mask = None if is_global else ring_mask

            hidden, k_out, v_out = layer(
                hidden, cos, sin,
                input_pos=input_pos, ple_signal=ple_signal,
                shared_k=shared_k, shared_v=shared_v,
                ring_indices=layer_ring_indices, ring_mask=layer_ring_mask,
            )

            if attn.store_full_length_kv:
                if is_global:
                    full_k, full_v = k_out, v_out
                else:
                    sliding_k, sliding_v = k_out, v_out

        hidden = self.norm(hidden)
        return hidden

    # ------------------------------------------------------------------
    # Split entry points for multimodal export
    # ------------------------------------------------------------------
    # The fused `forward` above does embed → PLE → layers → norm in one
    # call. Multimodal prefill needs a splice point between the word
    # embedding and the decoder so vision/audio encoder outputs can be
    # written into the placeholder positions. These two methods expose
    # that split. They require `use_ple=True` and a prior call to
    # `split_ple_for_export()`.

    def embed_for_export(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(inputs_embeds, ple_tok)` for the decoder to consume.

        `ple_tok` is the per-layer PLE token signal of shape (B, S, L, d),
        keyed on the raw input_ids. Gemma4's PLE token path fires at the
        placeholder slot regardless of whether the final embedding is text,
        vision, or audio — so the caller splices encoder outputs into
        `inputs_embeds` only, leaving `ple_tok` untouched.
        """
        assert self.config.use_ple, \
            "embed_for_export requires use_ple=True"
        assert self._ple_split, \
            "call split_ple_for_export() before embed_for_export"
        inputs_embeds = self.embed_tokens(input_ids)
        per_layer = [
            self._ple_per_layer_embs[i](input_ids)
            for i in range(self.config.num_hidden_layers)
        ]
        ple_tok = torch.stack(per_layer, dim=2)  # (B, S, L, d)
        return inputs_embeds, ple_tok

    def decode_from_embeds(
        self,
        inputs_embeds: torch.Tensor,   # (B, S, hidden)
        ple_tok: torch.Tensor,         # (B, S, L, d)
        input_pos: torch.Tensor,       # (S,)
    ) -> torch.Tensor:
        """Mirror of `forward` with embeddings supplied externally.

        PLE context is computed per-layer from `inputs_embeds` (which may
        contain spliced encoder outputs at placeholder positions); PLE
        token signal comes from `ple_tok` pre-computed in
        `embed_for_export`.
        """
        assert self.config.use_ple, \
            "decode_from_embeds requires use_ple=True"
        assert self._ple_split, \
            "call split_ple_for_export() before decode_from_embeds"

        B, S, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if self._use_kv_cache:
            cos_local = self._cos_local[input_pos]
            sin_local = self._sin_local[input_pos]
            cos_global = self._cos_global[input_pos]
            sin_global = self._sin_global[input_pos]
        else:
            cos_local, sin_local = self.rope_local(S, device)
            cos_global, sin_global = self.rope_global(S, device)

        # Lever F: pre-broadcast cos/sin once per forward so per-layer
        # apply_rotary_emb skips its unsqueeze block. Mirror of the
        # block in Gemma4EdgeModel.forward.
        if Gemma4Attention.use_prebroadcast_rope:
            if Gemma4Attention.use_bshd_kv:
                cos_local = cos_local.unsqueeze(0).unsqueeze(-2)
                sin_local = sin_local.unsqueeze(0).unsqueeze(-2)
                cos_global = cos_global.unsqueeze(0).unsqueeze(-2)
                sin_global = sin_global.unsqueeze(0).unsqueeze(-2)
            else:
                cos_local = cos_local.unsqueeze(0).unsqueeze(0)
                sin_local = sin_local.unsqueeze(0).unsqueeze(0)
                cos_global = cos_global.unsqueeze(0).unsqueeze(0)
                sin_global = sin_global.unsqueeze(0).unsqueeze(0)

        hidden = inputs_embeds
        hidden_for_ple = inputs_embeds

        # Lever B+C: compute ring indices + mask once per forward. All ring
        # layers consume the same values, so per-layer recomputation is
        # redundant. See ring_kv_cache.compute_shared_ring_state.
        ring_indices: Optional[torch.Tensor] = None
        ring_mask: Optional[torch.Tensor] = None
        if (
            Gemma4Attention.use_shared_ring_state
            and getattr(self, "_shared_ring_max_context_length", None) is not None
        ):
            from .transforms.kv_ring import compute_shared_ring_state
            ring_indices, ring_mask = compute_shared_ring_state(
                self._shared_ring_cache_positions,
                self._shared_ring_kv_cache_pos,
                input_pos,
                self._shared_ring_window,
                self._shared_ring_max_context_length,
                self._shared_ring_dtype,
            )

        sliding_k: Optional[torch.Tensor] = None
        sliding_v: Optional[torch.Tensor] = None
        full_k: Optional[torch.Tensor] = None
        full_v: Optional[torch.Tensor] = None

        for i, layer in enumerate(self.layers):
            is_global = self.layer_is_global[i]
            cos = cos_global if is_global else cos_local
            sin = sin_global if is_global else sin_local

            ple_tok_i = ple_tok[:, :, i, :]
            ple_ctx_i = self._ple_per_layer_projs[i](hidden_for_ple)
            ple_ctx_i = ple_ctx_i * (self.config.hidden_size ** -0.5)
            ple_ctx_i = self.per_layer_projection_norm(ple_ctx_i)
            ple_signal = (ple_ctx_i + ple_tok_i) * (2.0 ** -0.5)

            attn = layer.self_attn
            if attn.is_kv_shared_layer:
                if is_global:
                    shared_k, shared_v = full_k, full_v
                else:
                    shared_k, shared_v = sliding_k, sliding_v
            else:
                shared_k, shared_v = None, None

            # Only sliding (non-global) layers consume the precomputed ring
            # values; global layers keep their existing static-mask path.
            layer_ring_indices = None if is_global else ring_indices
            layer_ring_mask = None if is_global else ring_mask

            hidden, k_out, v_out = layer(
                hidden, cos, sin,
                input_pos=input_pos, ple_signal=ple_signal,
                shared_k=shared_k, shared_v=shared_v,
                ring_indices=layer_ring_indices, ring_mask=layer_ring_mask,
            )

            if attn.store_full_length_kv:
                if is_global:
                    full_k, full_v = k_out, v_out
                else:
                    sliding_k, sliding_v = k_out, v_out

        hidden = self.norm(hidden)
        return hidden


# ---------------------------------------------------------------------------
# Causal LM head — copied from reference Gemma4ForCausalLM
# ---------------------------------------------------------------------------

class Gemma4EdgeForCausalLM(nn.Module):
    """LM head with optional logit soft-capping and tied embeddings."""

    def __init__(self, config: Gemma4EdgeConfig):
        super().__init__()
        self.config = config
        self.model = Gemma4EdgeModel(config)

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Export-time knobs (set before torch.export). `return_last_only` slices
        # hidden to the final position before the lm_head — drops the logits
        # tensor from [B, S, vocab] to [B, 1, vocab], saving multi-GB of
        # activation arena at `seq_dim.max=2048`. `logits_dtype` casts the
        # logits + softcap pipeline to a narrower dtype.
        self.return_last_only: bool = False
        self.logits_dtype: torch.dtype = torch.float32

    def setup_caches(self, batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.float32) -> None:
        self.model.setup_caches(batch_size, max_seq_len, dtype)

    def reset_caches(self) -> None:
        self.model.reset_caches()

    def forward(self, input_ids: torch.Tensor, input_pos: Optional[torch.Tensor] = None):
        hidden = self.model(input_ids, input_pos)

        if self.return_last_only:
            hidden = hidden[:, -1:, :]

        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = F.linear(hidden, self.model.embed_tokens.weight)

        # Cast AFTER the lm_head matmul so the softcap intermediates (tanh,
        # cap * tanh, logits/cap) all run in the narrow dtype. Arena slots for
        # those intermediates then halve (fp32→fp16) or halve (fp32→bf16).
        if self.logits_dtype != logits.dtype:
            logits = logits.to(self.logits_dtype)

        # Final logit soft-capping (reference lines 1744-1747)
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)

        return logits

    # ------------------------------------------------------------------
    # Split entry points for multimodal export
    # ------------------------------------------------------------------
    # `embed_for_export` and `decode_from_embeds` combine to reproduce
    # `forward` while exposing a splice point for multimodal prefill.
    # Thin facades over Gemma4EdgeModel; see that class for semantics.

    def embed_for_export(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.embed_for_export(input_ids)

    def decode_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        ple_tok: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.model.decode_from_embeds(inputs_embeds, ple_tok, input_pos)

        if self.return_last_only:
            hidden = hidden[:, -1:, :]

        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = F.linear(hidden, self.model.embed_tokens.weight)

        if self.logits_dtype != logits.dtype:
            logits = logits.to(self.logits_dtype)

        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)

        return logits


# ---------------------------------------------------------------------------
# Export wrapper modules — thin nn.Module shims so torch.export sees the
# split entry points as each wrapper's `.forward`. Weights stay shared
# with the underlying Gemma4EdgeForCausalLM.
# ---------------------------------------------------------------------------

class Gemma4TokenEmbedderExport(nn.Module):
    """Wraps `Gemma4EdgeForCausalLM.embed_for_export` as a forward."""

    def __init__(self, lm: "Gemma4EdgeForCausalLM"):
        super().__init__()
        self.lm = lm

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lm.embed_for_export(input_ids)


class Gemma4TextDecoderExport(nn.Module):
    """Wraps `Gemma4EdgeForCausalLM.decode_from_embeds` as a forward."""

    def __init__(self, lm: "Gemma4EdgeForCausalLM"):
        super().__init__()
        self.lm = lm

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        ple_tok: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm.decode_from_embeds(inputs_embeds, ple_tok, input_pos)


# ---------------------------------------------------------------------------
# Quantization helpers (torchao 8da4w + weight-only int4)
# ---------------------------------------------------------------------------
#
# Remaining `torchao.dequantize_affine` ops on quantized embeddings (the ones
# XNNPACK doesn't delegate) are rewritten at to_executorch() time by setting
# ExecutorchBackendConfig.do_quant_fusion_and_const_prop=True. That triggers
# ExecuTorch's built-in QuantFusionPass + constant_prop_pass, which rewrites
# the `torchao.dequantize_affine + aten.embedding` pattern into
# `quantized_decomposed.embedding_{byte,4bit}` (portable kernels) and
# materialises packed int4 weights. No custom decomposition pass required.


def quantize_model(
    model: nn.Module,
    mode: str = "int4",
    group_size: int = 128,
    quantize_main_embeddings: bool = False,
    embedding_group_size: Optional[int] = None,
    quantize_ple_embeddings: bool = False,
    ple_embedding_group_size: Optional[int] = None,
    ple_embedding_bits: int = 4,
    dynamic_activation: bool = True,
    skip_linear_name_suffixes: Optional[tuple] = None,
) -> nn.Module:
    """Apply torchao quantization to model weights (and optionally activations).

    Linear layers use 8da4w (int8 dynamic activation + intX weight) by default,
    which enables proper XNNPACK delegation via XnnpackDynamicallyQuantizedPartitioner.
    Embeddings always use weight-only quantization (no activation quantization).

    Args:
        model: Model to quantize (modified in-place).
        mode: "int4" or "int8" — sets the weight dtype.
        group_size: Quantization group size for nn.Linear weights.
        quantize_main_embeddings: Quantize embed_tokens (main token embedding).
        embedding_group_size: Group size for main embedding (defaults to group_size).
        quantize_ple_embeddings: Quantize embed_tokens_per_layer (PLE embedding).
        ple_embedding_group_size: Group size for PLE embedding (defaults to group_size).
        ple_embedding_bits: Bit-width for PLE embeddings (2, 4, or 8). PLE projection
            linears always use the main weight_dtype since int2 GEMM has no kernel;
            only the lookup-dominated embedding benefits from int2.
        dynamic_activation: If True (default), use Int8DynamicActivationIntxWeightConfig
            for Linear layers (8da4w). If False, use IntxWeightOnlyConfig (weight-only).

    Returns:
        Quantized model (same object, modified in-place).
    """
    import torchao  # noqa: F401
    from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
    from torchao.quantization.granularity import PerGroup
    from torchao.utils import unwrap_tensor_subclass

    weight_dtype = torch.int4 if mode == "int4" else torch.int8

    _bits_to_dtype = {2: torch.int2, 4: torch.int4, 8: torch.int8}
    if ple_embedding_bits not in _bits_to_dtype:
        raise ValueError(
            f"ple_embedding_bits must be 2, 4, or 8 (got {ple_embedding_bits})"
        )
    ple_weight_dtype = _bits_to_dtype[ple_embedding_bits]

    if embedding_group_size is None:
        embedding_group_size = group_size
    if ple_embedding_group_size is None:
        ple_embedding_group_size = group_size

    def _make_embedding_config(gs, dtype=None):
        return IntxWeightOnlyConfig(
            weight_dtype=dtype or weight_dtype, granularity=PerGroup(gs)
        )

    def _make_linear_config(gs):
        if dynamic_activation:
            from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig
            return Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=PerGroup(gs),
            )
        return IntxWeightOnlyConfig(weight_dtype=weight_dtype, granularity=PerGroup(gs))

    def _linear_filter(m, fqn):
        if not isinstance(m, nn.Linear):
            return False
        # per_layer_input_gate (hidden_size → ple_dim) is multiplied into every
        # layer's residual, so quantization error compounds 35× across layers.
        # Real-weights experiments showed 8da4w on this gate alone
        # destroys coherence; keeping it in fp16 restores proper end-of-turn
        # generation. The output projection and other Linears quantize fine.
        if "per_layer_input_gate" in fqn:
            return False
        if skip_linear_name_suffixes and any(
            fqn.endswith(sfx) for sfx in skip_linear_name_suffixes
        ):
            return False
        # PLE projection linears: only exclude when --quantize-ple is set
        # (then _ple_proj_filter handles them weight-only). Otherwise, let them
        # flow through 8da4w so XNNPACK doesn't fall back to its f32-SME kernel
        # path which faults on CPUs without SME (Apple M1/M2).
        if "_ple_per_layer" in fqn and quantize_ple_embeddings:
            return False
        in_feat = m.weight.shape[1]
        return in_feat >= group_size and in_feat % group_size == 0

    def _main_embedding_filter(m, fqn):
        if not isinstance(m, nn.Embedding):
            return False
        if "per_layer" in fqn:
            return False
        embed_dim = m.weight.shape[1]
        return embed_dim >= embedding_group_size and embed_dim % embedding_group_size == 0

    def _ple_emb_filter(m, fqn):
        """Match split PLE embeddings (applies configurable bit-width)."""
        if "_ple_per_layer_embs" not in fqn:
            return False
        if not isinstance(m, nn.Embedding):
            return False
        dim = m.weight.shape[1]
        return dim >= ple_embedding_group_size and dim % ple_embedding_group_size == 0

    def _ple_proj_filter(m, fqn):
        """Match split PLE projections (kept at main weight_dtype — int2 GEMM not supported)."""
        if "_ple_per_layer_projs" not in fqn:
            return False
        if not isinstance(m, nn.Linear):
            return False
        dim = m.weight.shape[0]
        return dim >= ple_embedding_group_size and dim % ple_embedding_group_size == 0

    with torch.no_grad():
        # Weight-tying trap: when tie_word_embeddings=True, the lm_head path
        # is `F.linear(hidden, embed_tokens.weight)` — no `nn.Linear` module,
        # so _linear_filter can't quantize it. If we then weight-only quantize
        # `embed_tokens`, the same `dequantize_affine` op feeds BOTH the
        # embedding lookup AND the tied linear. QuantFusionPass rewrites the
        # embedding use but leaves the linear use dangling, and to_executorch
        # rejects it for lacking an `.out` variant. Fix: untie by giving the
        # model a real lm_head Linear (cloned from the embedding) so 8da4w
        # picks it up and XNNPACK delegates it cleanly.
        should_untie = (
            quantize_main_embeddings
            and hasattr(model, "lm_head")
            and model.lm_head is None
            and getattr(model, "config", None) is not None
            and getattr(model.config, "tie_word_embeddings", False)
        )
        if should_untie:
            cfg = model.config
            lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            lm_head.weight = nn.Parameter(
                model.model.embed_tokens.weight.detach().clone()
            )
            model.lm_head = lm_head
            model.config.tie_word_embeddings = False

        quantize_(model, _make_linear_config(group_size), filter_fn=_linear_filter)

        if quantize_main_embeddings:
            quantize_(model, _make_embedding_config(embedding_group_size),
                      filter_fn=_main_embedding_filter)

        if quantize_ple_embeddings:
            quantize_(
                model,
                _make_embedding_config(ple_embedding_group_size, dtype=ple_weight_dtype),
                filter_fn=_ple_emb_filter,
            )
            quantize_(
                model,
                _make_embedding_config(ple_embedding_group_size),
                filter_fn=_ple_proj_filter,
            )

        model = unwrap_tensor_subclass(model)

    return model


# ---------------------------------------------------------------------------
# Weight loading from SafeTensors
# ---------------------------------------------------------------------------

def _config_from_json(config_json_path: str) -> Gemma4EdgeConfig:
    """Build Gemma4EdgeConfig from a HuggingFace config.json."""
    import json
    with open(config_json_path) as f:
        raw = json.load(f)
    tc = raw.get("text_config", {})
    return Gemma4EdgeConfig(
        vocab_size=tc.get("vocab_size", 262_144),
        hidden_size=tc.get("hidden_size", 1536),
        num_hidden_layers=tc.get("num_hidden_layers", 35),
        num_attention_heads=tc.get("num_attention_heads", 8),
        num_key_value_heads=tc.get("num_key_value_heads", 1),
        head_dim=tc.get("head_dim", 256),
        global_head_dim=tc.get("global_head_dim", 512),
        intermediate_size=tc.get("intermediate_size", 6144),
        ple_dim=tc.get("hidden_size_per_layer_input", 256),
        ple_vocab_size=tc.get("vocab_size_per_layer_input", 262_144),
        use_ple=tc.get("hidden_size_per_layer_input", 0) > 0,
        sliding_window=tc.get("sliding_window", 512),
        max_position_embeddings=tc.get("max_position_embeddings", 131072),
        layer_types=tc.get("layer_types", None),
        rope_theta=tc.get("rope_parameters", {}).get("full_attention", {}).get("rope_theta", 1_000_000.0),
        rope_local_base_freq=tc.get("rope_parameters", {}).get("sliding_attention", {}).get("rope_theta", 10_000.0),
        partial_rotary_factor=tc.get("rope_parameters", {}).get("full_attention", {}).get("partial_rotary_factor", 0.25),
        num_kv_shared_layers=tc.get("num_kv_shared_layers", 20),
        use_double_wide_mlp=tc.get("use_double_wide_mlp", True),
        final_logit_softcapping=tc.get("final_logit_softcapping", 30.0),
        tie_word_embeddings=raw.get("tie_word_embeddings", True),
    )


def load_from_safetensors(
    safetensors_path: str,
    config: Optional[Gemma4EdgeConfig] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Gemma4EdgeForCausalLM:
    """Load Gemma 4 text decoder weights from a SafeTensors checkpoint.

    Key mapping: model.language_model.X -> model.X
    Auto-reads config.json from the same directory if no config provided.
    """
    from safetensors import safe_open
    import os

    if config is None:
        # Try to read config.json from same directory, parent, or cwd
        search_dirs = [
            os.path.dirname(os.path.abspath(safetensors_path)),
            os.path.dirname(os.path.dirname(os.path.abspath(safetensors_path))),
            os.getcwd(),
        ]
        config_json = None
        for d in search_dirs:
            candidate = os.path.join(d, "config.json")
            if os.path.exists(candidate):
                config_json = candidate
                break
        if config_json:
            print(f"  Reading config from {config_json}")
            config = _config_from_json(config_json)
        else:
            config = Gemma4EdgeConfig()

    model = Gemma4EdgeForCausalLM(config)

    state_dict = {}
    prefix = "model.language_model."
    with safe_open(safetensors_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                new_key = "model." + key[len(prefix):]
                state_dict[new_key] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    expected_missing = set()
    if config.tie_word_embeddings:
        expected_missing.add("lm_head.weight")

    real_missing = [k for k in missing if k not in expected_missing]
    if real_missing:
        print(f"WARNING: Missing keys: {real_missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")

    return model.to(dtype)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def gemma4_config() -> Gemma4EdgeConfig:
    """Default config matching the official Gemma 4 checkpoint."""
    return Gemma4EdgeConfig()
