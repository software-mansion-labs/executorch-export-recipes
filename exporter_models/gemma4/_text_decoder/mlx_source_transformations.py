# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLX source transformations for the Gemma 4 E2B text decoder.

Replaces the attention internals with MLX custom ops that lower to optimized
Metal kernels:

- ``torch.ops.mlx.rope`` for rotary position embeddings
- ``torch.ops.mlx.kv_cache_update`` (via the MLX KV cache modules) for cache scatter
- ``torch.ops.mlx.custom_sdpa`` for GQA-aware scaled dot-product attention

Only ``Gemma4Attention`` is rewritten. The self/cross decoder and transformer
forwards are left untouched so the YOCO KV-sharing (donor/consumer layers) and
per-layer-embedding plumbing keep working exactly as in the eager/XNNPACK path:
the rewritten attention still honors the ``shared_kv`` input and returns
``(attn_output, kv_to_share)``.
"""

import types
from typing import Optional, Tuple

import executorch.backends.mlx.custom_ops  # noqa: F401 — registers mlx:: ops
import torch
import torch.nn as nn
from executorch.backends.mlx.llm.cache import (
    KVCache as MLXKVCache,
    RingBufferKVCache as MLXRingKVCache,
)


def _ring_sliding_window_mask(
    start_pos: int,
    seq_len: int,
    window_size: int,
    buffer_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive ring-buffer sliding-window mask [1, 1, seq_len, buffer_size].

    Stateless reconstruction of the slot→position mapping from start_pos and
    buffer_size (mirrors RingBufferKVCache.create_sliding_window_mask), so it
    works for consumer layers that hold no cache object.
    """
    w = window_size
    b = buffer_size
    end_pos = start_pos + seq_len

    slots = torch.arange(b, dtype=torch.long)
    last_write_slot = (end_pos - 1) % b
    current_cycle_base = end_pos - 1 - last_write_slot
    pos_current = current_cycle_base + slots
    pos_previous = current_cycle_base - b + slots
    cache_pos = torch.where(slots <= last_write_slot, pos_current, pos_previous)

    pos_q = (start_pos + torch.arange(seq_len, dtype=torch.long)).view(-1, 1)
    delta = pos_q - cache_pos
    attn_mask = (cache_pos >= 0) & (delta >= 0) & (delta < w)

    zero = torch.zeros(1, dtype=dtype)
    neg_inf = torch.full((1,), float("-inf"), dtype=dtype)
    return torch.where(attn_mask, zero, neg_inf).unsqueeze(0).unsqueeze(0)


def _permute_partial_rope_channels(attn: nn.Module) -> None:
    """Reorder q/k head channels so partial RoPE maps onto mlx::rope.

    Gemma4's full-attention layers use partial RoPE with zero-padded
    inv_freq: rotate_half pairs channel ``i`` with ``i + head_dim/2`` across
    the FULL head_dim, rotating only the first ``rotary_dim/2`` pairs.
    mlx::rope (traditional=False, dims=rotary_dim) instead pairs ``j`` with
    ``j + rotary_dim/2`` within the first rotary_dim channels. The two are
    related by a fixed channel permutation — and SDPA's q·k dot product is
    invariant when the SAME permutation is applied to both q and k. So we
    permute the q/k projection rows (and their per-channel norm weights)
    once at export and use the fused rope kernel everywhere. v and o_proj
    are untouched (attention output channels don't move).

    Frequencies: gemma computes inv_freq[j] = theta^(-2j/head_dim); mlx
    derives theta'^(-2j/dims). theta' = theta^(rotary_dim/head_dim) makes
    them equal exactly (see _mlx_rope).
    """
    D = attn.head_dim
    H = D // 2
    R = attn.rotary_dim // 2
    if attn.rotary_dim >= D:
        return  # full-dim rope: pairing already matches mlx::rope

    perm = (
        list(range(0, R))            # rotating firsts
        + list(range(H, H + R))      # their rotate_half partners
        + list(range(R, H))          # pass-through (first half remainder)
        + list(range(H + R, D))      # pass-through (second half remainder)
    )
    perm_t = torch.tensor(perm, dtype=torch.long)

    with torch.no_grad():
        for proj, n_heads in (
            (attn.q_proj, attn.num_heads),
            (attn.k_proj, attn.num_kv_heads),
        ):
            w = proj.weight
            proj.weight.copy_(
                w.view(n_heads, D, -1)[:, perm_t, :].reshape(w.shape)
            )
            if proj.bias is not None:
                b = proj.bias
                proj.bias.copy_(b.view(n_heads, D)[:, perm_t].reshape(b.shape))
        for norm in (attn.q_norm, attn.k_norm):
            w = getattr(norm, "weight", None)
            if w is not None:
                w.copy_(w[perm_t])

    attn._mlx_rope_permuted = True


def _replace_attention_forward(attn: nn.Module) -> None:
    """Rewrite a Gemma4Attention.forward to use MLX custom ops.

    Mirrors the eager forward's YOCO contract: consumes ``shared_kv`` on
    shared layers (rotating Q only), updates the MLX KV cache on non-shared
    layers, and returns the donor (k, v) so later layers can reuse them.
    """

    def _mlx_forward(
        self,
        hidden_states: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = hidden_states.shape
        if self._mlx_compile_friendly:
            # Positions stay tensors end-to-end: no .item() means no
            # data-dependent scalars in the lowered graph, so the whole
            # method can be wrapped in mx::compile at runtime.
            start_pos = input_pos[0]
        else:
            start_pos = input_pos[0].item()

        q = self.q_proj(hidden_states)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        if self.is_kv_shared_layer and shared_kv is not None:
            # YOCO consumer: reuse the donor's cached K/V (already RoPE'd by the
            # donor), rotate Q only — exactly the eager path.
            k_cache, v_cache = shared_kv
            q = self._mlx_rope(q, start_pos, input_pos, T)
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            k = self.k_norm(k)
            v = self.v_norm(v)

            q = self._mlx_rope(q, start_pos, input_pos, T)
            k = self._mlx_rope(k, start_pos, input_pos, T)

            k_cache, v_cache = self.kv_cache.update(input_pos, k, v)

        # Donor layers share their cached K/V (post-update) with consumers.
        kv_to_share: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.is_kv_donor_layer:
            kv_to_share = (k_cache, v_cache)

        if self.is_sliding:
            # Sliding layers use the ring buffer with proper windowed eviction.
            # The causal-slice fast path is ONLY valid while positions stay within
            # one window; once start_pos exceeds sliding_window the ring wraps and
            # a naive causal slice attends to scrambled/out-of-window positions
            # (catastrophic divergence past the window — long conversations). Use
            # the ring-window mask, which is correct at every position.
            buffer_size = 2 * self._mlx_sliding_window
            cache = self._mlx_mask_cache
            key = ("ring", id(input_pos))
            if cache.get("_key") != key[1]:
                cache.clear()
                cache["_key"] = key[1]
            sdpa_mask = cache.get(key)
            if sdpa_mask is None:
                sdpa_mask = _ring_sliding_window_mask(
                    start_pos, T, self._mlx_sliding_window, buffer_size,
                    k_cache.dtype
                )
                cache[key] = sdpa_mask
            y = torch.ops.mlx.custom_sdpa(
                q,
                k_cache,
                v_cache,
                start_pos=buffer_size - T,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
        elif self._mlx_compile_friendly:
            # Static-shape variant: attend over the FULL cache with a causal
            # mask built from the position tensor (unwritten slots are blocked
            # by the mask). start_pos = ctx_len - T makes custom_sdpa's
            # internal [0:stop] slice a static full-length no-op.
            ctx_len = self._mlx_ctx_len
            cache = self._mlx_mask_cache
            ckey = ("causal", id(input_pos))
            if cache.get("_key") != ckey[1]:
                cache.clear()
                cache["_key"] = ckey[1]
            causal_mask = cache.get(ckey)
            if causal_mask is None:
                cache_pos = torch.arange(ctx_len, dtype=torch.long)
                q_pos = (start_pos + torch.arange(T, dtype=torch.long)).view(-1, 1)
                allowed = cache_pos <= q_pos
                zero = torch.zeros(1, dtype=k_cache.dtype)
                neg_inf = torch.full((1,), float("-inf"), dtype=k_cache.dtype)
                causal_mask = (
                    torch.where(allowed, zero, neg_inf).unsqueeze(0).unsqueeze(0)
                )
                cache[ckey] = causal_mask
            y = torch.ops.mlx.custom_sdpa(
                q,
                k_cache,
                v_cache,
                start_pos=ctx_len - T,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
        else:
            # Full-attention layers legitimately attend to all prior positions;
            # custom_sdpa's internal [0:stop_pos] causal slice is correct here.
            y = torch.ops.mlx.custom_sdpa(
                q,
                k_cache,
                v_cache,
                start_pos=start_pos,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scaling,
            )

        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(y), kv_to_share

    def _mlx_rope(self, x: torch.Tensor, start_pos, input_pos, T) -> torch.Tensor:
        """Apply RoPE matching the model's native HF convention.

        Sliding layers rotate the full head_dim — mlx::rope (split-half) matches
        the native rotate_half exactly (verified cos=1.0), so use the fast Metal
        kernel. Full-attention layers use *partial* RoPE where the native code
        zero-pads inv_freq to head_dim//2 and rotate_half pairs dim i with dim
        i+head_dim/2 across the FULL head_dim — a pairing mlx::rope cannot express
        (it pairs within rotary_dim and passes the rest through). So for full
        layers, replicate the native rotate_half with plain ops (MLX-delegatable),
        using the same zero-padded cos/sin the model computes."""
        if self.is_sliding:
            if self._mlx_compile_friendly:
                return torch.ops.mlx.rope_t(
                    x, self.head_dim, start_pos, False, self.rope_theta, 1.0, None
                )
            return torch.ops.mlx.rope(
                x, self.head_dim, start_pos, False, self.rope_theta, 1.0, None
            )
        if getattr(self, "_mlx_rope_permuted", False):
            # Channels were permuted at export so the partial-RoPE pairing
            # matches mlx::rope's. theta' = theta^(rot/D) reproduces gemma's
            # head_dim-normalized frequencies under mlx's dims-normalized
            # exponent: theta'^(-2j/rot) == theta^(-2j/D).
            theta_eff = float(self.rope_theta) ** (self.rotary_dim / self.head_dim)
            if self._mlx_compile_friendly:
                return torch.ops.mlx.rope_t(
                    x, self.rotary_dim, start_pos, False, theta_eff, 1.0, None
                )
            return torch.ops.mlx.rope(
                x, self.rotary_dim, start_pos, False, theta_eff, 1.0, None
            )
        # Full-attention partial RoPE: native HF path. self.inv_freq is already
        # zero-padded to head_dim//2 (real freqs then zeros), so cos/sin built
        # from it leave the non-rotary dims unchanged.
        freqs_cos, freqs_sin = self._get_rope_freqs(input_pos, T)
        return self._apply_rope_single(x, freqs_cos, freqs_sin)

    attn._mlx_rope = types.MethodType(_mlx_rope, attn)
    attn.forward = types.MethodType(_mlx_forward, attn)


def mlx_source_transformations(
    model: nn.Module,
    dtype: torch.dtype = torch.float32,
    compile_friendly: bool = False,
) -> None:
    """Apply MLX source transformations to a Gemma 4 E2B text decoder in-place.

    Gives EVERY attention its own MLX KV cache (ring buffer for sliding layers,
    linear for full-attention layers) and rewrites the attention forward to use
    ``mlx.rope`` / ``mlx.custom_sdpa``.

    YOCO donor-reuse is disabled for MLX: the donor's ``mlx.kv_cache_update``
    buffer mutation cannot be shared across the consumers' separate MLX delegate
    partitions. This is numerically faithful — the checkpoint ships real
    ``k_proj``/``v_proj`` weights for all 35 layers (YOCO is a runtime
    recompute-avoidance optimization, not weight sharing), so each consumer
    recomputes the same K/V the donor would have. The self/cross decoder and
    transformer forwards are untouched, so per-layer embeddings keep working.
    """
    config = model.config

    # Shared per-forward mask cache. The ring and causal masks depend only on
    # (start_pos, T) — identical across all 35 layers in a single forward — so
    # they are built once by the first layer that needs each and reused by the
    # rest. Under mx::compile the trace records the mask subgraph once instead
    # of 35 times, cutting ~600 redundant kernel dispatches per token. Keyed by
    # id(input_pos) so a new token's position tensor invalidates the cache.
    mask_cache: dict = {}

    for module in model.modules():
        if type(module).__name__ != "Gemma4Attention":
            continue
        attn = module

        # Needed by sliding layers to build the ring mask.
        attn._mlx_sliding_window = config.sliding_window
        attn._mlx_compile_friendly = compile_friendly
        attn._mlx_ctx_len = config.max_seq_len
        attn._mlx_mask_cache = mask_cache

        # Full-attention layers: permute q/k channels so partial RoPE runs on
        # the fused mlx::rope kernel instead of decomposed cos/sin ops.
        if not attn.is_sliding:
            _permute_partial_rope_channels(attn)

        # Only non-shared layers own a cache; YOCO consumers reuse the donor's.
        if not attn.is_kv_shared_layer:
            if attn.is_sliding:
                attn.kv_cache = MLXRingKVCache(
                    max_batch_size=1,
                    max_context_length=config.sliding_window,
                    n_heads=attn.num_kv_heads,
                    head_dim=attn.head_dim,
                    dtype=dtype,
                )
            else:
                attn.kv_cache = MLXKVCache(
                    max_batch_size=1,
                    max_context_length=config.max_seq_len,
                    n_heads=attn.num_kv_heads,
                    head_dim=attn.head_dim,
                    enable_dynamic_shape=True,
                    dtype=dtype,
                )
            attn.kv_cache.use_index_copy = compile_friendly

        _replace_attention_forward(attn)
