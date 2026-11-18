"""
Ring buffer KV caches for Gemma 4 sliding-window attention layers.

The default Gemma `KVCache` pre-allocates `[B, H, max_seq_len, D]` for every
layer, including sliding-window layers that only need to attend to
`sliding_window` tokens. This module provides ring variants that allocate
`[B, H, 2*window, D]` (or `[B, 2*window, H, D]` for the quantized variant)
and overwrite slots modulo `2*window` as new tokens arrive.

Why 2*window — verbose explanation in
`examples/models/llama/attention.py:241`. Short version: when prefilling a
chunk of length `S` into a window of `W`, you need `2W` slots so the chunk
can still attend to the previous `W` tokens without self-eviction.

Implementation notes
--------------------
* Float ring (`Gemma4RingKVCache`) — direct mirror of Gemma's `KVCache`
  with `[B, H, 2W, D]` storage and modular `index_copy_` writes.

* Quantized ring (`Gemma4QuantizedRingKVCache`) — inherits from the
  sideloaded Llama `QuantizedKVCache` to reuse `_quantize` and the four
  per-token cache buffers (k/v + scales + zero_points). We do NOT inherit
  from Llama's `QuantizedRingKVCache`: that class always passes `indices`
  to the parent `update()`, but the parent's non-custom-op write path
  hard-asserts `indices is None`, and the custom-op path needs
  `torch.ops.llama.update_cache_with_indices` which we can't register
  without a C++ kernel build (out of scope). So we override `update()`
  and write the int8 cache slots ourselves with `index_copy_`.

* Mask handling — the dispatch in `Gemma4Attention.forward` checks
  `getattr(self.kv_cache, "is_ring_buffer", False)` and calls
  `make_attention_mask(input_pos)` instead of indexing the precomputed
  `[max_seq_len, max_seq_len]` `self.mask` buffer. The swap function in
  this file also clears that buffer (replaces it with a scalar) on
  sliding-window layers, since the ring path makes it dead weight — and
  on long contexts that buffer dwarfs the KV cache savings.
"""
import torch
import torch.nn as nn

from .kv_quant import (
    Gemma4QuantizedKVCache,
    QuantizedCacheType,
    QuantizedKVCache,
    _get_gemma_kv_cache_class,
)


# ---------------------------------------------------------------------------
# Helpers — mirror examples/models/llama/attention.py:155 and :186
# ---------------------------------------------------------------------------

def _ring_causal_mask(
    cache_positions: torch.Tensor,
    window_size: int,
    start_pos: int,
    seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a `[seq_len, 2*window]` causal+window mask for the ring buffer.

    `cache_positions[s]` holds the absolute position of the token currently
    stored in slot `s`, or -1 if the slot has never been written. A slot
    is attendable iff:
      - it is valid (cache_positions[s] >= 0)
      - the query is at or after the key in absolute order (delta >= 0)
      - the key is within the sliding window (delta < window_size)
    """
    pos_q = start_pos + torch.arange(seq_len, dtype=torch.long).view(-1, 1)
    delta = pos_q - cache_positions
    valid = (cache_positions >= 0) & (delta >= 0) & (delta < window_size)
    zero = torch.tensor(0.0, dtype=dtype)
    neginf = torch.tensor(float("-inf"), dtype=dtype)
    return torch.where(valid, zero, neginf)


def _calc_ring_indices(
    cache_positions: torch.Tensor,
    start_pos: int,
    seq_len: int,
    max_context_length: int,
) -> torch.Tensor:
    """Compute the modular write indices for a ring buffer of length
    `max_context_length`, AND in-place update `cache_positions` so that
    every slot records the absolute position currently stored in it
    (with -1 sentinel for never-written slots).

    Mirrors `CachePositionsManager.calculate_positions_and_update_indices`
    from `examples/models/llama/attention.py:186`.
    """
    orig_indices = torch.arange(seq_len, dtype=torch.long) + start_pos
    indices = orig_indices % max_context_length

    # Slots that haven't been written yet (positions >= start_pos in the
    # arange-of-2W slots) get reset to -1 sentinel; the index_copy_ below
    # then stamps the freshly-written absolute positions into the touched
    # slots. This handles the very first write (start_pos == 0) where the
    # rest of the buffer must remain invalid.
    full_t = torch.full((max_context_length,), -1, dtype=torch.long)
    arange_t = torch.arange(max_context_length, dtype=torch.long)
    new_positions = torch.where(arange_t < start_pos, cache_positions, full_t)
    cache_positions.copy_(new_positions)
    cache_positions.index_copy_(0, indices, orig_indices)

    return indices


def compute_shared_ring_state(
    cache_positions: torch.Tensor,
    kv_cache_pos: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int,
    max_context_length: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (indices, attention_mask) once for a forward pass; mutates
    `cache_positions` and advances `kv_cache_pos`.

    Replaces the per-layer `_calc_ring_indices` + `_ring_causal_mask` calls
    when `Gemma4Attention.use_shared_ring_state` is True. Math is identical
    to running both helpers in sequence inside each ring cache's update(),
    but here it runs once and the result is broadcast to every ring layer
    in the model — saving (N_ring - 1) duplications of the index + mask
    computation per forward.

    Also drops the dead `where(arange<start_pos, cache_positions, -1)`
    block from the original `_calc_ring_indices` (validated bit-identical).
    """
    # The caller may rewind after a tool call or reset a session without
    # clearing native buffers. Its explicit position is authoritative; an
    # independently advancing counter would write into the wrong ring slots.
    start_pos = input_pos[0].item()
    torch._check_is_size(start_pos)
    seq_len = input_pos.shape[0]

    # indices (Lever A simplified — no dead `where` block)
    orig_indices = torch.arange(seq_len, dtype=torch.long) + start_pos
    indices = orig_indices % max_context_length
    cache_positions.index_copy_(0, indices, orig_indices)

    # attention mask, reads cache_positions AFTER the stamp above
    mask = _ring_causal_mask(
        cache_positions, window_size, start_pos, seq_len, dtype=dtype,
    )

    kv_cache_pos.add_(seq_len)
    return indices, mask


# ---------------------------------------------------------------------------
# Float ring KV cache — `[B, H, 2W, D]` storage matching Gemma's KVCache.
# ---------------------------------------------------------------------------

class Gemma4RingKVCache(nn.Module):
    """Float ring buffer for sliding-window layers.

    Same `update(k_val, v_val)` signature as `gemma4.KVCache`, returns the
    full `[B, H, 2W, D]` cache. The caller computes the attention mask
    via `make_attention_mask(input_pos)` instead of indexing the static
    `[max_seq_len, max_seq_len]` mask buffer.
    """

    is_ring_buffer = True

    def __init__(
        self,
        batch_size: int,
        window_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # Local import to avoid circular dep at module load.
        from ..text import Gemma4Attention
        self.window_size = window_size
        self.max_context_length = 2 * window_size
        self.cache_fp_type = dtype
        # Storage layout mirrors `Gemma4Attention.use_bshd_kv`. Default BHSD
        # matches the Gemma reference; BSHD matches custom_sdpa expectations
        # and skips the per-step transpose+contiguous round-trip.
        self._use_bshd = Gemma4Attention.use_bshd_kv
        if self._use_bshd:
            cache_shape = (batch_size, self.max_context_length, num_kv_heads, head_dim)
        else:
            cache_shape = (batch_size, num_kv_heads, self.max_context_length, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False,
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False,
        )
        self.register_buffer(
            "cache_positions",
            torch.full((self.max_context_length,), -1, dtype=torch.long),
            persistent=False,
        )
        # Same name & semantics as Gemma's KVCache so the existing
        # `InitializedMutableBufferPass(["kv_cache_pos"])` picks it up.
        # We only ever read element 0 (`start_pos`); the array length
        # follows the Gemma convention (= cache size).
        self.register_buffer(
            "kv_cache_pos",
            torch.arange(0, self.max_context_length),
            persistent=False,
        )

    def update(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        input_pos: torch.Tensor,
        precomputed_indices: "torch.Tensor | None" = None,
    ):
        # k_val/v_val layout matches `use_bshd_kv`:
        #   BHSD (default): [B, H, S, D], write along dim 2.
        #   BSHD:           [B, S, H, D], write along dim 1.
        # `input_pos` carries the SYMBOLIC seq-len dim from the exported graph;
        # use it instead of k_val.shape[...] for the ring-index computation to
        # avoid specializing the dim.
        if precomputed_indices is not None:
            # Shared-state path (Lever B+C): position bookkeeping happened
            # once at model level; we just consume the indices and skip
            # the per-layer kv_cache_pos advance (also model-level now).
            indices = precomputed_indices
        else:
            seq_len = input_pos.shape[0]
            start_pos = input_pos[0].item()
            torch._check_is_size(start_pos)
            indices = _calc_ring_indices(
                self.cache_positions, start_pos, seq_len, self.max_context_length,
            )
            self.kv_cache_pos.add_(seq_len)
        s_dim = 1 if self._use_bshd else 2
        self.k_cache.index_copy_(s_dim, indices, k_val)
        self.v_cache.index_copy_(s_dim, indices, v_val)
        return self.k_cache, self.v_cache

    def make_attention_mask(
        self,
        input_pos: torch.Tensor,
        precomputed_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Return a `[seq_len, 2*window]` mask aligned to the current ring
        contents. MUST be called AFTER `update()` so `cache_positions`
        reflects the writes for the current step.
        """
        if precomputed_mask is not None:
            return precomputed_mask
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        seq_len = input_pos.shape[0]
        return _ring_causal_mask(
            self.cache_positions, self.window_size, start_pos, seq_len,
            dtype=self.cache_fp_type,
        )

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_positions.fill_(-1)
        self.kv_cache_pos.copy_(torch.arange(0, self.max_context_length))


# ---------------------------------------------------------------------------
# Quantized ring KV cache — int8 ring on top of Llama's per-token quantize.
# ---------------------------------------------------------------------------

class Gemma4QuantizedRingKVCache(QuantizedKVCache):
    """int8 ring buffer for sliding-window layers.

    Storage layout is inherited from Llama's `QuantizedKVCache`:
        k_cache, v_cache              : [B, 2W, H, D] int8
        k_cache_scales, v_*_scales    : [B, 2W, H, 1] float32
        k_cache_zero_points, v_*_zps  : [B, 2W, H, 1] int8

    The `update()` method takes `[B, H, S, D]` (Gemma's convention),
    transposes to `[B, S, H, D]` for storage, quantizes per-token via the
    parent's `_quantize` helper, writes to ring slots with `index_copy_`,
    dequantizes the full cache, overlays the latest float values at the
    freshly-written slots (matches the parent's float-overlay pattern),
    and returns the result transposed back to `[B, H, 2W, D]`.

    Class-level flag `use_packed_quant_artifacts` (default False) is the
    Lever D switch: it drops the float overlay (lines marked
    "float overlay" below). The current step's K/V then comes from the
    dequantized int8 cache rather than the raw float — saves 2 host
    `index_put` ops per ring layer per forward at the cost of one token's
    worth of int8 quantization error. Gated OFF for backward compat with
    the earlier recipes, which were bit-identical against eager fp16.
    """

    is_ring_buffer = True

    # Lever D — when True, skip the float overlay onto the just-written
    # slots of the dequantized K/V cache. The current step then attends
    # over the lossy (int8-asymmetric-dequantized) values for the freshly
    # written tokens, instead of the original float values. Validated
    # against eager.
    use_packed_quant_artifacts = False

    def __init__(
        self,
        batch_size: int,
        window_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            max_batch_size=batch_size,
            max_context_length=2 * window_size,
            n_heads=num_kv_heads,
            head_dim=head_dim,
            cache_type=QuantizedCacheType.AffineAsymmetric,
            use_custom_update_cache_op=False,  # we do manual index_copy_ writes
            return_float_values=True,
        )
        # Local import to avoid circular dep at module load.
        from ..text import Gemma4Attention
        self._use_bshd = Gemma4Attention.use_bshd_kv
        # Match Gemma's surrounding model dtype so dequantized k/v line up
        # with `q` in the matmul that follows.
        self.cache_fp_type = dtype
        self.window_size = window_size
        # max_context_length is set by parent to 2 * window_size.

        self.register_buffer(
            "cache_positions",
            torch.full((self.max_context_length,), -1, dtype=torch.long),
            persistent=False,
        )
        # `kv_cache_pos` for InitializedMutableBufferPass and start_pos source.
        self.register_buffer(
            "kv_cache_pos",
            torch.arange(0, self.max_context_length),
            persistent=False,
        )

    def _quantize(self, value):
        # ExecuTorch's choose_qparams_per_token_asymmetric kernel only
        # supports float32 input (op_choose_qparams.cpp:50). Cast up from
        # float16 before quantizing.
        return super()._quantize(value.float())

    def update(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        input_pos: torch.Tensor,
        precomputed_indices: "torch.Tensor | None" = None,
    ):
        # Storage is [B, 2W, H, D] (parent BSHD). In BSHD mode the caller
        # already hands us BSHD — skip the transpose round-trip. In BHSD
        # mode we transpose in/out to match storage.
        if self._use_bshd:
            k_val_bshd = k_val
            v_val_bshd = v_val
        else:
            k_val_bshd = k_val.transpose(1, 2)
            v_val_bshd = v_val.transpose(1, 2)

        if precomputed_indices is not None:
            # Shared-state path: indices + cache_positions updates + kv_cache_pos
            # advance all done once at model level. Just consume.
            indices = precomputed_indices
            advance_pos = False
        else:
            # Take seq_len from input_pos (symbolic dim) rather than from
            # k_val_bshd.shape[1] — same reason as the float ring variant.
            seq_len = input_pos.shape[0]
            start_pos = input_pos[0].item()
            torch._check_is_size(start_pos)
            indices = _calc_ring_indices(
                self.cache_positions, start_pos, seq_len, self.max_context_length,
            )
            advance_pos = True

        # Per-token int8 quantize (parent helper). Returns shapes:
        #   q_k:      [B, S, H, D] int8
        #   k_scales: [B, S, H, 1] (whatever choose_qparams returns; we cast)
        #   k_zps:    [B, S, H, 1] (ditto)
        q_k, k_scales, k_zps = self._quantize(k_val_bshd)
        q_v, v_scales, v_zps = self._quantize(v_val_bshd)

        k_scales = k_scales.to(torch.float32)
        v_scales = v_scales.to(torch.float32)
        k_zps = k_zps.to(self.quantized_cache_dtype)
        v_zps = v_zps.to(self.quantized_cache_dtype)

        # Manual ring writes via index_copy_ along dim 1 (S).
        self.k_cache.index_copy_(1, indices, q_k)
        self.v_cache.index_copy_(1, indices, q_v)
        self.k_cache_scales.index_copy_(1, indices, k_scales)
        self.v_cache_scales.index_copy_(1, indices, v_scales)
        self.k_cache_zero_points.index_copy_(1, indices, k_zps)
        self.v_cache_zero_points.index_copy_(1, indices, v_zps)

        # Dequantize the full int8 cache to float, then overlay the latest
        # float values at the freshly-written slots so the current step
        # uses unquantized k/v (matches the parent's float-overlay pattern
        # in `_update_and_return_float_values`).
        k_out = torch.ops.quantized_decomposed.dequantize_per_token(
            self.k_cache,
            self.k_cache_scales.to(torch.float64),
            self.k_cache_zero_points.to(torch.int64),
            torch.iinfo(self.quantized_cache_dtype).min,
            torch.iinfo(self.quantized_cache_dtype).max,
            self.quantized_cache_dtype,
            self.cache_fp_type,
        )
        v_out = torch.ops.quantized_decomposed.dequantize_per_token(
            self.v_cache,
            self.v_cache_scales.to(torch.float64),
            self.v_cache_zero_points.to(torch.int64),
            torch.iinfo(self.quantized_cache_dtype).min,
            torch.iinfo(self.quantized_cache_dtype).max,
            self.quantized_cache_dtype,
            self.cache_fp_type,
        )
        # Float overlay onto freshly-written slots — restores per-token
        # quantization error for the current step's K/V contribution.
        # Lever D (use_packed_quant_artifacts=True) skips these two
        # `index_put` ops to absorb 2 host ops per ring layer per forward.
        # The current step then attends over the dequantized (lossy)
        # values for the new tokens; accuracy impact validated at
        # cos vs ref > 0.9995.
        if not Gemma4QuantizedRingKVCache.use_packed_quant_artifacts:
            k_out.index_copy_(1, indices, k_val_bshd)
            v_out.index_copy_(1, indices, v_val_bshd)

        if advance_pos:
            self.kv_cache_pos.add_(seq_len)
        if self._use_bshd:
            # Stay in BSHD — caller wants [B, 2W, H, D] for custom_sdpa.
            return k_out, v_out
        # Back to Gemma's [B, H, 2W, D] for the attention matmul.
        return k_out.transpose(1, 2), v_out.transpose(1, 2)

    def make_attention_mask(
        self,
        input_pos: torch.Tensor,
        precomputed_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        if precomputed_mask is not None:
            return precomputed_mask
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        seq_len = input_pos.shape[0]
        return _ring_causal_mask(
            self.cache_positions, self.window_size, start_pos, seq_len,
            dtype=self.cache_fp_type,
        )

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_cache_scales.fill_(1.0)
        self.v_cache_scales.fill_(1.0)
        self.k_cache_zero_points.zero_()
        self.v_cache_zero_points.zero_()
        self.cache_positions.fill_(-1)
        self.kv_cache_pos.copy_(torch.arange(0, self.max_context_length))


# ---------------------------------------------------------------------------
# Swap function — install ring caches on every sliding-window attention.
# ---------------------------------------------------------------------------

def replace_sliding_kv_cache_with_ring_kv_cache(model: nn.Module) -> nn.Module:
    """Walk the model, find every sliding-window `Gemma4Attention` (i.e.
    `is_global=False`) and swap its KV cache for the ring variant.

    * If the existing cache is a `Gemma4QuantizedKVCache`, install a
      `Gemma4QuantizedRingKVCache` (so quantize-kv-cache + ring-kv-cache
      compose correctly when both flags are set).
    * If it's the float `KVCache`, install a `Gemma4RingKVCache`.
    * If `2*window >= existing cache size`, the ring would be the same
      size or larger — skip the swap (no memory win).

    Also clears the precomputed `[max_seq_len, max_seq_len]` static mask
    buffer on the swapped attention modules: the ring path computes the
    mask dynamically per step, so the static buffer is dead weight — and
    on long contexts it dominates KV cache memory.

    Must run AFTER `model.setup_caches(...)`. If a quantized-KV-cache swap
    is also being applied, run that first so this function picks up the
    `Gemma4QuantizedKVCache` instances and produces quantized rings.
    """
    # Local import to avoid circular dep at module load.
    from ..text import Gemma4Attention
    Gemma4KVCache = _get_gemma_kv_cache_class()

    swapped = 0
    skipped_too_small = 0
    for module in model.modules():
        if not isinstance(module, Gemma4Attention) or module.is_global:
            continue
        cache = module.kv_cache
        if cache is None:
            continue
        window = module.config.sliding_window
        ring_size = 2 * window

        if isinstance(cache, Gemma4QuantizedKVCache):
            # Quantized storage is always [B, S, H, D] (Llama convention),
            # regardless of Gemma4Attention.use_bshd_kv — that flag only
            # affects whether the adapter's update() transposes in/out.
            B, S, H, D = cache.k_cache.shape
            dtype = cache.cache_fp_type
            ring_cls = Gemma4QuantizedRingKVCache
        elif isinstance(cache, Gemma4KVCache):
            # Float Gemma cache storage follows `use_bshd_kv`:
            #   BHSD: [B, H, S, D]    BSHD: [B, S, H, D].
            if Gemma4Attention.use_bshd_kv:
                B, S, H, D = cache.k_cache.shape
            else:
                B, H, S, D = cache.k_cache.shape
            dtype = cache.k_cache.dtype
            ring_cls = Gemma4RingKVCache
        else:
            continue

        if ring_size >= S:
            # Existing cache is already <= 2W slots; ring would not save
            # memory (and would actually grow it). Leave it alone.
            skipped_too_small += 1
            continue

        module.kv_cache = ring_cls(
            batch_size=B, window_size=window,
            num_kv_heads=H, head_dim=D, dtype=dtype,
        )

        # Drop the static mask buffer — the forward dispatch will call
        # `kv_cache.make_attention_mask(input_pos)` instead. Replacing
        # it with `None` keeps `self.mask` as a valid attribute access
        # but releases the `[max_seq_len, max_seq_len]` storage.
        if "mask" in module._buffers:
            del module._buffers["mask"]
        module.mask = None

        swapped += 1

    print(
        f"  Ring KV cache: swapped {swapped} sliding-window layer(s)"
        + (f", skipped {skipped_too_small} (cache already <= 2W)" if skipped_too_small else "")
    )
    return model
