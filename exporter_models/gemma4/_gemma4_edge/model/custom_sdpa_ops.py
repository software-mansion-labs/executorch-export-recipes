# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Eager Python reference impls for the two Gemma4-flavored Vulkan SDPA ops
# defined in PLAN_gemma4_custom_sdpa_shader.md (§1.2):
#
#   et_vk::gemma_sdpa_with_kv_cache  — fused (update + sdpa)
#   et_vk::gemma_custom_sdpa         — read-only SDPA against an
#                                      already-updated cache
#
# These are CPU-only reference implementations used to validate the math
# of the new ops at G1/G1b before any Vulkan shader work. Both ops:
#   - hardcode scale = 1.0 (Gemma uses QKV-norm)
#   - accept an optional additive 4D mask of shape (1, 1, S_q, kv_len)
#     OR (1, H_q, S_q, kv_len) (per-head, audio path)
#   - support the mask=None / is_causal=True path for global layers
#   - optional tanh-softcap (audio path): softcap_value > 0 applies
#     `tanh((attn + mask) / softcap_value) * softcap_value` BEFORE the
#     softmax (Option B: combined mask, softcap-after-add). For audio's
#     cap=50, masked positions saturate to ~-cap which is negligible in
#     the softmax denominator vs unmasked logits in [-cap, +cap].
#
# Layout assumptions match §1.4 of the plan:
#   query : (B=1, S, H_q,  D)         BSHD
#   key   : (B=1, S, H_kv, D)         BSHD       (only fused op)
#   value : (B=1, S, H_kv, D)         BSHD       (only fused op)
#   k_cache, v_cache: (1, max_context_len, H_kv, D)   BSHD
#   attn_mask: (1, 1, S, kv_len) OR (1, H_q, S, kv_len) additive float
#              (-inf or finfo.min masked, 0 else; per-head encodes
#               Transformer-XL rel-pos bias for audio)

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Library + schemas
# ---------------------------------------------------------------------------

_LIB_NAME = "et_vk"
_lib = torch.library.Library(_LIB_NAME, "FRAGMENT")


def _maybe_define(schema: str, name: str) -> None:
    """Idempotent define so re-importing this module in the same process
    (e.g. across notebook cells / pytest sessions) is harmless."""
    op_ns = getattr(torch.ops, _LIB_NAME)
    if hasattr(op_ns, name):
        return
    _lib.define(schema)


_maybe_define(
    "gemma_sdpa_with_kv_cache(Tensor query, Tensor key, Tensor value, "
    "Tensor(a!) key_cache, Tensor(b!) value_cache, "
    "SymInt start_pos, SymInt seq_len, "
    "Tensor? attn_mask=None, bool is_causal=False, "
    "float softcap_value=0.0) -> Tensor",
    "gemma_sdpa_with_kv_cache",
)

_maybe_define(
    "gemma_custom_sdpa(Tensor query, Tensor key_cache, Tensor value_cache, "
    "SymInt start_pos, SymInt seq_len, "
    "Tensor? attn_mask=None, bool is_causal=False, "
    "float softcap_value=0.0) -> Tensor",
    "gemma_custom_sdpa",
)


# ---------------------------------------------------------------------------
# Reference math
# ---------------------------------------------------------------------------

def _sdpa_reference(
    query: torch.Tensor,         # (B, S, H_q, D) BSHD
    key_full: torch.Tensor,      # (B, L, H_kv, D) BSHD — already includes new K
    value_full: torch.Tensor,    # (B, L, H_kv, D) BSHD
    start_pos: int,
    seq_len: int,
    attn_mask,                   # None, (1, 1, S, kv_len), or (1, H_q, S, kv_len) additive float
    is_causal: bool,
    softcap_value: float = 0.0,  # > 0: apply `tanh(x/cap)*cap` AFTER mask add, BEFORE softmax
) -> torch.Tensor:
    """Compute SDPA with scale=1.0 against the full cache, then return
    output in the query's BSHD layout.

    softcap_value > 0 applies `tanh(x / cap) * cap` after mask add and
    before softmax. This is Option B (combined mask + softcap-after-add)
    -- mathematically not bit-equivalent to HF's `softcap(QK^T+matrix_bd)
    + pad_mask` ordering, but the divergence is negligible because masked
    positions saturate to ~-cap which is dominated by exp(unmasked_max)
    in the softmax denominator (analysis: for cap=50, masked weight ~
    exp(-cap - L_max) ~ exp(-100)).
    """
    B, S, H_q, D = query.shape
    _, L, H_kv, _ = key_full.shape
    assert B == 1, "Plan §1.4 requires batch=1."
    assert S == seq_len, f"seq_len mismatch: {S} vs {seq_len}"

    # The plan keeps GQA broadcast inside the shader (use_broadcast_gqa).
    # For the reference, we explicitly repeat K/V to H_q heads.
    group = H_q // H_kv
    if group > 1:
        key_full = key_full.repeat_interleave(group, dim=2)
        value_full = value_full.repeat_interleave(group, dim=2)

    # Move to BHSD for the matmul.
    q = query.transpose(1, 2)          # (1, H_q, S,  D)
    k = key_full.transpose(1, 2)       # (1, H_q, L,  D)
    v = value_full.transpose(1, 2)     # (1, H_q, L,  D)

    # scale = 1.0 (no multiply).
    attn = torch.matmul(q, k.transpose(-2, -1))  # (1, H_q, S, L)

    if attn_mask is not None:
        # attn_mask: (1, 1, S, kv_len) broadcast OR (1, H_q, S, kv_len) per-head.
        # Slice to L if longer along kv dim.
        m = attn_mask
        if m.size(-1) != L:
            assert m.size(-1) >= L, (
                f"attn_mask kv-dim {m.size(-1)} < cache len {L}"
            )
            m = m[..., :L]
        # Cast to attn dtype; broadcast-add handles dim[1]=1 or H_q.
        attn = attn + m.to(attn.dtype)
    elif is_causal:
        # mask out positions in [start_pos+1+s, L) for query row s.
        device = attn.device
        col = torch.arange(L, device=device).view(1, 1, 1, L)
        row = torch.arange(S, device=device).view(1, 1, S, 1) + start_pos
        causal = col > row
        attn = attn.masked_fill(causal, float("-inf"))
    # else: no mask — caller asserts nothing to mask (rare; keep behavior
    # symmetric to llama's path).

    # Tanh-softcap (audio path: cap=50). Applied AFTER mask add per
    # Option B. Math diff from HF (which applies softcap BEFORE pad-mask
    # add) is negligible because saturated -cap mask values are still
    # exp(-L_max - cap) away from unmasked logits in the softmax.
    if softcap_value > 0.0:
        attn = torch.tanh(attn / softcap_value) * softcap_value

    # Match Gemma reference: softmax in fp32, cast back to query dtype.
    probs = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v)        # (1, H_q, S, D)
    return out.transpose(1, 2).contiguous()   # (1, S, H_q, D)


# ---------------------------------------------------------------------------
# Fused op: update cache then sdpa
# ---------------------------------------------------------------------------

@torch.library.impl(_lib, "gemma_sdpa_with_kv_cache", "CompositeExplicitAutograd")
def _gemma_sdpa_with_kv_cache_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    start_pos: int,
    seq_len: int,
    attn_mask=None,
    is_causal: bool = False,
    softcap_value: float = 0.0,
):
    # In-place write at [:, start_pos:start_pos+seq_len, :, :] (BSHD cache).
    end = start_pos + seq_len
    key_cache[:, start_pos:end, :, :] = key
    value_cache[:, start_pos:end, :, :] = value
    return _sdpa_reference(
        query, key_cache, value_cache, start_pos, seq_len, attn_mask, is_causal,
        softcap_value=softcap_value,
    )


@torch.library.impl(_lib, "gemma_sdpa_with_kv_cache", "Meta")
def _gemma_sdpa_with_kv_cache_meta(
    query, key, value, key_cache, value_cache,
    start_pos, seq_len, attn_mask=None, is_causal=False,
    softcap_value: float = 0.0,
):
    return torch.empty_like(query)


# ---------------------------------------------------------------------------
# Read-only op: SDPA against an already-updated cache (KV-consumer layers)
# ---------------------------------------------------------------------------

@torch.library.impl(_lib, "gemma_custom_sdpa", "CompositeExplicitAutograd")
def _gemma_custom_sdpa_impl(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    start_pos: int,
    seq_len: int,
    attn_mask=None,
    is_causal: bool = False,
    softcap_value: float = 0.0,
):
    return _sdpa_reference(
        query, key_cache, value_cache, start_pos, seq_len, attn_mask, is_causal,
        softcap_value=softcap_value,
    )


@torch.library.impl(_lib, "gemma_custom_sdpa", "Meta")
def _gemma_custom_sdpa_meta(
    query, key_cache, value_cache, start_pos, seq_len,
    attn_mask=None, is_causal=False,
    softcap_value: float = 0.0,
):
    return torch.empty_like(query)
