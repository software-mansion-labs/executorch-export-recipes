"""Source transform: rewrite gemma4 RoPE to use the fused Vulkan op.

`torch.ops.et_vk.apply_rotary_emb(q, k, cos, sin)` is registered with an
eager CompositeExplicitAutograd impl that runs the llama-form rotation. So
we don't need to rely on the FX pattern matcher in
`backends/vulkan/patterns/rope.py` — calling the op directly produces a
single FX node that the Vulkan partitioner accepts as-is.

What this transform does (per non-shared Gemma4Attention instance, after
`replace_gemma_sdpa_with_custom_op` has installed the SDPA forward):

  1. Permute q, k from HF head_dim layout `[r..., i...]`
     to llama interleaved `[r0, i0, r1, i1, ...]` (pure view+transpose+flatten).
  2. Slice cos/sin from `(seq, head_dim)` to `(seq, head_dim/2)`.
  3. Call the Vulkan-fused op: `torch.ops.et_vk.apply_rotary_emb(q', k', cos_h, sin_h)`.
  4. Permute outputs back to HF layout.

For consumer (shared-KV) layers, only Q is rotated (no K projection); the
fused op requires both q AND k, so consumers stay on the un-fused single-
tensor `apply_rotary_emb` path. Acceptable: the transform's purpose is to
recover llama-shape on the layers that compute their own KV.

Eager parity is bit-exact (validated in iter37a).

Usage:
    replace_gemma_sdpa_with_custom_op(model)        # SDPA fused op
    replace_gemma_rope_with_paired_fused(model)     # then rope fused op
"""
from __future__ import annotations
from types import MethodType
from typing import Optional

import torch
import torch.nn as nn

# Side-effect import: registers torch.ops.et_vk.apply_rotary_emb (and friends).
import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

import gemma4.model.text as g4


# ---------------------------------------------------------------------------
# Layout helpers (pure view+transpose; no gather)
# ---------------------------------------------------------------------------

def _hf_to_llama_layout(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    return x.reshape(*x.shape[:-1], 2, half).transpose(-1, -2).flatten(-2)


def _llama_to_hf_layout(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    return x.reshape(*x.shape[:-1], half, 2).transpose(-1, -2).flatten(-2)


def _paired_fused_rope(
    q: torch.Tensor,        # (B, S, Hq, D) HF layout
    k: torch.Tensor,        # (B, S, Hk, D) HF layout
    cos: torch.Tensor,      # (seq, D), HF cat([f, f])
    sin: torch.Tensor,
):
    """Call the Vulkan-fused RoPE op on paired q/k. Returns (q', k') in HF
    layout (the surrounding model code expects HF layout)."""
    d = cos.shape[-1]
    half = d // 2
    cos_h = cos[..., :half]
    sin_h = sin[..., :half]

    # Slice cos/sin to current seq length to match llama's pattern body
    # (which doesn't rely on the seq dim being inferred).
    S = q.size(1)
    cos_h = cos_h[:S]
    sin_h = sin_h[:S]

    q_l = _hf_to_llama_layout(q)
    k_l = _hf_to_llama_layout(k)
    q_rot, k_rot = torch.ops.et_vk.apply_rotary_emb(q_l, k_l, cos_h, sin_h)
    return _llama_to_hf_layout(q_rot), _llama_to_hf_layout(k_rot)


# ---------------------------------------------------------------------------
# Per-instance forward replacement (composes with replace_gemma_sdpa_with_custom_op)
# ---------------------------------------------------------------------------

def _make_paired_rope_standalone_forward():
    """Forward for non-shared Gemma4Attention: paired-fused RoPE + fused SDPA."""

    def new_forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        shared_k: Optional[torch.Tensor] = None,
        shared_v: Optional[torch.Tensor] = None,
        # Gemma4DecoderLayer.forward (gemma4.py:945) unconditionally threads
        # these through; accept and ignore — the paired-rope+fused-SDPA path
        # builds its own sliding mask, so the precomputed ring state is N/A.
        # Required for dynamo arg-binding even when shared-ring-state is off
        # (the kwargs are None in that case).
        ring_indices: Optional[torch.Tensor] = None,
        ring_mask: Optional[torch.Tensor] = None,
    ):
        from gemma4.model.transforms.sdpa import _build_mask4d_for_sliding

        attn_t = type(self)
        if not getattr(attn_t, "use_bshd_kv", False):
            raise RuntimeError("paired-rope transform requires use_bshd_kv=True")
        assert input_pos is not None

        B, S, _ = hidden.shape

        # Q/K projections + norms (mirrors gemma4.Gemma4Attention.forward 503-518)
        q = self.q_proj(hidden).view(B, S, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # PAIRED FUSED ROPE — single Vulkan op call
        q, k = _paired_fused_rope(q, k, cos, sin)

        start_pos_arg = input_pos[0].item()
        if self.is_global:
            mask4d = None
            is_causal = True
        else:
            mask4d = _build_mask4d_for_sliding(self, input_pos, q.dtype)
            is_causal = False

        k_cache = self.kv_cache.k_cache
        v_cache = self.kv_cache.v_cache
        out_bshd = torch.ops.et_vk.gemma_sdpa_with_kv_cache(
            q, k, v,
            k_cache, v_cache,
            start_pos_arg, S,
            mask4d, is_causal,
        )
        out = out_bshd.reshape(B, S, -1)
        return self.o_proj(out), k, v

    return new_forward


def replace_gemma_rope_with_paired_fused(model: nn.Module) -> nn.Module:
    """Replace each non-shared Gemma4Attention's forward (already SDPA-fused)
    with one using paired fused-RoPE. Mutates model in place. Idempotent."""
    from gemma4.model.text import Gemma4Attention

    n_replaced = 0
    n_consumer_skipped = 0
    new_factory = _make_paired_rope_standalone_forward()
    for module in model.modules():
        if not isinstance(module, Gemma4Attention):
            continue
        # iter118: use `is_kv_shared_layer` instead of the stale
        # `kv_cache is None and kv_source is not None` test. Per iter65,
        # every layer (including consumers) gets its own KVCache, so the
        # old check always evaluated False and consumer layers were
        # incorrectly treated as standalone (their forward got the paired
        # fused-RoPE that does its OWN k_proj — wrong for consumers,
        # which must read shared K from the source layer).
        is_consumer = bool(getattr(module, "is_kv_shared_layer", False))
        if is_consumer:
            n_consumer_skipped += 1
            continue
        if getattr(module, "_paired_rope_installed", False):
            continue
        module.forward = MethodType(new_factory, module)
        module._paired_rope_installed = True
        n_replaced += 1
    model._paired_rope_stats = {
        "replaced": n_replaced,
        "consumer_skipped": n_consumer_skipped,
    }
    return model
