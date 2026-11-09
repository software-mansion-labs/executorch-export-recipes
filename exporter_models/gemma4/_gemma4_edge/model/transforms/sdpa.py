"""Source transform: rewrite Gemma4Attention.forward to emit the two
new Vulkan ops `et_vk::gemma_sdpa_with_kv_cache` and
`et_vk::gemma_custom_sdpa`.

Plan reference: PLAN_gemma4_custom_sdpa_shader.md §5.

Classifier:
  - Standalone        : self.kv_cache is not None and self.kv_source is None
  - KV-source         : self.kv_cache is not None and self.kv_source is None
                        — same Python signature as Standalone for our
                        purposes; both emit the FUSED op.
                        (A "source" is just a non-shared layer that
                        another later layer happens to alias; from the
                        layer's own forward perspective, it is
                        indistinguishable from Standalone.)
  - KV-consumer       : self.kv_cache is None and self.kv_source is not None
                        → emits the READ-ONLY op against
                        self.kv_source.kv_cache.

Mask (sliding layers): we materialize the additive 4-D float mask
(1, 1, S, max_seq_len), force float dtype, .contiguous().
For global layers we pass mask=None, is_causal=True (per plan §5 step 3).

This module purposely does NOT touch:
  - the apply_rotary_emb path
  - the q_norm / k_norm / v_norm path
  - the o_proj / output reshape
  - the per-layer use_broadcast_gqa / use_dynamic_kv_slice / ring KV
    paths — those are out of scope for the v0 source transform; if they
    are set on a model when this transform runs, we raise. Long-term they
    can be folded in as the C++ kernels gain capability.
"""

from __future__ import annotations

from typing import Optional

import math
import os as _os

import torch
import torch.nn as nn

# Side-effect: defines torch.ops.et_vk.gemma_{sdpa_with_kv_cache,custom_sdpa}.
from .. import custom_sdpa_ops  # noqa: F401


# When set, replace `self.q_norm(q)` / `self.k_norm(k)` with a fp32-internal
# RMSNorm: cast input to fp32, compute mean/rsqrt/scale in fp32 (incl.
# fp32 weights), cast result back to input dtype. v_norm is intentionally
# left alone (different module on V projection).
_FP32_QK_NORM = _os.environ.get("FP32_QK_NORM", "0") == "1"


def _fp32_rmsnorm_inline(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute RMSNorm in fp32 with fp32 weights, return result in x's
    input dtype."""
    orig_dtype = x.dtype
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    var = x_fp32.pow(2).mean(-1, keepdim=True)
    x_fp32 = x_fp32 * torch.rsqrt(var + eps)
    x_fp32 = x_fp32 * weight_fp32
    return x_fp32.to(orig_dtype)

# Local imports kept lazy (avoid pulling in heavy gemma4 imports at module
# load time for callers that only want the function symbol).


def _build_mask4d_for_sliding(
    attn_module: nn.Module,
    input_pos: torch.Tensor,
    q_dtype: torch.dtype,
) -> torch.Tensor:
    """Build (1, 1, S, max_seq_len) additive float mask, .contiguous(),
    dtype matching query.

    Mirrors the static-mask path in `Gemma4Attention.forward` (gemma4.py
    `:540-543` and `:578-583`), reduced to the sliding-layer case
    (we never call this for global layers).
    """
    # Static buffer path: self.mask is the 2D (max_seq_len, max_seq_len)
    # mask. Sliced rows give (S, max_seq_len). Bool → additive on the fly.
    use_bool_mask = getattr(type(attn_module), "use_bool_mask", False)
    mask_row = attn_module.mask[input_pos]  # (S, max_seq_len)
    if use_bool_mask:
        mask_row = torch.zeros_like(
            mask_row, dtype=q_dtype,
        ).masked_fill(mask_row, float("-inf"))
    else:
        mask_row = mask_row.to(q_dtype)
    mask4d = mask_row.unsqueeze(0).unsqueeze(0).contiguous()
    return mask4d


def _make_standalone_or_source_forward(orig_forward):
    """Build a replacement forward for a non-shared (Standalone or
    KV-source) Gemma4Attention. Emits the FUSED op."""

    def new_forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        shared_k: Optional[torch.Tensor] = None,
        shared_v: Optional[torch.Tensor] = None,
        # Gemma4DecoderLayer.forward (gemma4.py:945) unconditionally threads
        # these through; accept and ignore — the fused-SDPA path builds its
        # own sliding mask, so the precomputed ring state is N/A. Required
        # for dynamo arg-binding even when shared-ring-state is off (the
        # kwargs are None in that case).
        ring_indices: Optional[torch.Tensor] = None,
        ring_mask: Optional[torch.Tensor] = None,
    ):
        # Guard: the v0 transform does not yet handle these knobs.
        attn_t = type(self)
        if getattr(attn_t, "use_bshd_kv", False) is False:
            raise RuntimeError(
                "gemma_sdpa_source_transform requires Gemma4Attention.use_bshd_kv=True "
                "(BSHD layout). Set it on the class before applying the transform."
            )
        if getattr(attn_t, "use_dynamic_kv_slice", False):
            raise RuntimeError(
                "use_dynamic_kv_slice is not yet supported by the new Vulkan ops; "
                "disable it before applying gemma_sdpa_source_transform."
            )
        if getattr(self, "use_broadcast_gqa", False):
            # Broadcast-GQA is fine — the new op handles GQA internally.
            pass
        if not self._use_cache:
            raise RuntimeError(
                "gemma_sdpa_source_transform requires KV cache. Call setup_caches() first."
            )
        assert input_pos is not None, "KV cache update requires input_pos"
        # Ring KV is out of scope.
        if getattr(self.kv_cache, "is_ring_buffer", False):
            raise RuntimeError(
                "Ring KV cache not supported by gemma_sdpa_source_transform v0."
            )

        from ..text import apply_rotary_emb  # local import

        B, S, _ = hidden.shape

        # Q
        q = self.q_proj(hidden).view(B, S, self.num_heads, self.head_dim)
        if _FP32_QK_NORM:
            q = _fp32_rmsnorm_inline(q, self.q_norm.weight, self.q_norm.eps)
        else:
            q = self.q_norm(q)
        # use_bshd_kv is asserted True above → no transpose.
        q = apply_rotary_emb(q, cos, sin)

        # K/V (this layer is non-shared so we always project)
        k = self.k_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
        if _FP32_QK_NORM:
            k = _fp32_rmsnorm_inline(k, self.k_norm.weight, self.k_norm.eps)
        else:
            k = self.k_norm(k)
        v = self.v_norm(v)
        k = apply_rotary_emb(k, cos, sin)

        # SymInt at trace time (export converts the .item() of an int64
        # tensor to a SymInt); int at eager time.
        start_pos_arg = input_pos[0].item()

        if self.is_global:
            mask4d = None
            is_causal = True
        else:
            mask4d = _build_mask4d_for_sliding(self, input_pos, q.dtype)
            is_causal = False

        # The new op operates on the layer's own KV cache (KVCache module).
        # Both Standalone and KV-source emit the fused op; KV-source's
        # cache is later read by consumer layers via self.kv_cache.
        k_cache = self.kv_cache.k_cache
        v_cache = self.kv_cache.v_cache

        out_bshd = torch.ops.et_vk.gemma_sdpa_with_kv_cache(
            q, k, v,
            k_cache, v_cache,
            start_pos_arg, S,
            mask4d, is_causal,
        )

        out = out_bshd.reshape(B, S, -1)
        # Return (out, k, v) so the outer model can thread shared_k/v
        # to consumer layers (existing contract from gemma4.py).
        return self.o_proj(out), k, v

    return new_forward


def _make_consumer_forward(orig_forward):
    """Build a replacement forward for a KV-consumer Gemma4Attention
    (shared layer). Emits the READ-ONLY op against the source's cache."""

    def new_forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        shared_k: Optional[torch.Tensor] = None,
        shared_v: Optional[torch.Tensor] = None,
        # Gemma4DecoderLayer.forward (gemma4.py:945) unconditionally threads
        # these through; accept and ignore — the consumer-SDPA path reads
        # the source layer's KV cache directly and the precomputed ring
        # state is N/A. Required for dynamo arg-binding.
        ring_indices: Optional[torch.Tensor] = None,
        ring_mask: Optional[torch.Tensor] = None,
    ):
        attn_t = type(self)
        if getattr(attn_t, "use_bshd_kv", False) is False:
            raise RuntimeError("requires use_bshd_kv=True")
        if getattr(attn_t, "use_dynamic_kv_slice", False):
            raise RuntimeError("dynamic_kv_slice unsupported in v0")
        if not self._use_cache:
            raise RuntimeError("setup_caches() must be called first")
        assert input_pos is not None
        assert self.kv_source is not None, (
            "KV-consumer classifier mismatch: kv_source must be set"
        )
        if getattr(self.kv_source.kv_cache, "is_ring_buffer", False):
            raise RuntimeError("ring KV cache not supported by v0")

        from ..text import apply_rotary_emb

        B, S, _ = hidden.shape

        q = self.q_proj(hidden).view(B, S, self.num_heads, self.head_dim)
        if _FP32_QK_NORM:
            q = _fp32_rmsnorm_inline(q, self.q_norm.weight, self.q_norm.eps)
        else:
            q = self.q_norm(q)
        q = apply_rotary_emb(q, cos, sin)

        if self.is_global:
            mask4d = None
            is_causal = True
        else:
            mask4d = _build_mask4d_for_sliding(self, input_pos, q.dtype)
            is_causal = False

        # Instead of calling the read-only gemma_custom_sdpa,
        # call the writer op gemma_sdpa_with_kv_cache against the source's
        # cache, feeding the SAME k/v that the source already wrote (via
        # shared_k/shared_v). The two ops then chain through torch.export's
        # buffer-mutation functionalization (cache_after_source ->
        # cache_after_consumer), so the consumer reads from the same
        # post-mutation graph value the source produced. The previous
        # gemma_custom_sdpa path created a separate graph node for the
        # cache that the partitioner left uninitialized — symptom: first
        # generated token plausible, then <pad>×N.
        assert shared_k is not None and shared_v is not None, (
            "consumer forward expects shared_k/shared_v threaded from the "
            "preceding source layer"
        )
        # Write into the consumer's OWN cache (each layer now has
        # its own KVCache; see gemma4.py setup_caches). Eliminates the
        # cross-partition shared-buffer aliasing the Vulkan partitioner
        # introduced when source and consumer pointed at one cache buffer.
        k_cache = self.kv_cache.k_cache
        v_cache = self.kv_cache.v_cache
        start_pos_arg = input_pos[0].item()

        out_bshd = torch.ops.et_vk.gemma_sdpa_with_kv_cache(
            q, shared_k, shared_v,
            k_cache, v_cache,
            start_pos_arg, S,
            mask4d, is_causal,
        )

        out = out_bshd.reshape(B, S, -1)
        # Consumer layers don't produce new K/V — pass through what the
        # source produced (shared_k/shared_v inputs).
        return self.o_proj(out), shared_k, shared_v

    return new_forward


def _classify(attn_module: nn.Module) -> str:
    """Return 'standalone', 'consumer'. (Source/Standalone are
    indistinguishable from the layer's own forward perspective and
    receive the same rewrite.)

    The consumer test switched from
        `kv_cache is None and kv_source is not None`
    to
        `is_kv_shared_layer`
    because the per-layer-KVCache fix gave every layer its own
    `kv_cache`, so the old `kv_cache is None` test always returned False
    and the consumer code path was dead. `is_kv_shared_layer` is the
    canonical truth set in `Gemma4Attention.__init__` (gemma4.py L374-378).
    """
    is_consumer = bool(getattr(attn_module, "is_kv_shared_layer", False))
    if is_consumer:
        # Sanity: a consumer layer must have kv_source wired.
        assert getattr(attn_module, "kv_source", None) is not None, (
            "is_kv_shared_layer=True but kv_source is None — wiring bug"
        )
        return "consumer"
    # Either standalone or source — both run the fused op.
    return "standalone"


def replace_gemma_sdpa_with_custom_op(model: nn.Module, layer_filter=None) -> nn.Module:  # noqa: E501
    """Walk the model and rewrite each Gemma4Attention's forward to emit
    the new Vulkan ops. Returns the model (mutated in place).

    `layer_filter`: optional callable(layer_idx:int) -> bool. If provided,
    only layers where it returns True are rewritten; the rest keep their
    original SDPA forward. Used for layer-count ablations. layer_idx is the
    enumeration order in `model.modules()` of Gemma4Attention instances
    (== layer index for standard Gemma4 architectures).

    Counts attached as `model._gemma_sdpa_transform_stats = {...}`.
    """
    from ..text import Gemma4Attention

    n_standalone = 0
    n_consumer = 0
    n_skipped = 0
    layer_idx = -1
    for mod in model.modules():
        if not isinstance(mod, Gemma4Attention):
            continue
        layer_idx += 1
        if layer_filter is not None and not layer_filter(layer_idx):
            n_skipped += 1
            continue
        kind = _classify(mod)
        if kind == "consumer":
            new_fwd = _make_consumer_forward(mod.forward)
            n_consumer += 1
        else:
            new_fwd = _make_standalone_or_source_forward(mod.forward)
            n_standalone += 1
        # Per-instance method override: bind to mod so `self` is correct.
        mod.forward = new_fwd.__get__(mod, type(mod))

    model._gemma_sdpa_transform_stats = {
        "standalone_or_source": n_standalone,
        "consumer": n_consumer,
        "skipped": n_skipped,
    }
    return model
