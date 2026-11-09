"""
Quantized KV cache for Gemma 4 — thin adapter around the Llama implementation
at `executorch/examples/models/llama/source_transformation/custom_kv_cache.py`.

The Gemma4 `KVCache.update(k_val, v_val)` signature differs from Llama's
`update(input_pos, k_val, v_val)`: Gemma tracks position implicitly via the
`kv_cache_pos` buffer. The `Gemma4QuantizedKVCache` adapter slices that
buffer to derive `input_pos` and forwards to the parent class.

Historical note: the original `gemma_export/quantized_kv_cache.py` sideloaded
this module with `importlib.util.spec_from_file_location`, reading the file
straight off disk at a path derived from `__file__`, after fabricating four
fake `sys.modules` entries to bypass a parent `__init__` that used to fail to
import. That is no longer necessary — the installed ExecuTorch wheel ships
`examples/models/llama/...` as a real importable package. The sideload was
also actively harmful: it bypassed the installed module in favour of a
different copy on disk, so the two could silently diverge, and it hard-bound
the package to a fixed position in the ExecuTorch source tree.
"""
import torch
import torch.nn as nn

from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    QuantizedCacheType,
    QuantizedKVCache,
)

__all__ = [
    "QuantizedCacheType",
    "QuantizedKVCache",
    "Gemma4QuantizedKVCache",
    "replace_kv_cache_with_quantized_kv_cache",
]


# ---------------------------------------------------------------------------
# Gemma4 adapter
# ---------------------------------------------------------------------------

# Imported lazily to avoid a circular import (gemma4.py is the model).
def _get_gemma_kv_cache_class():
    from ..text import KVCache
    return KVCache


class Gemma4QuantizedKVCache(QuantizedKVCache):
    """Quantized KV cache that matches Gemma4 `KVCache`'s constructor and
    `update(k_val, v_val)` signature, while reusing Llama's int8 storage,
    quantize/dequantize logic, and float-fallback for the latest position.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            max_batch_size=batch_size,
            max_context_length=max_seq_len,
            n_heads=num_kv_heads,
            head_dim=head_dim,
            cache_type=QuantizedCacheType.AffineAsymmetric,
            use_custom_update_cache_op=False,  # don't depend on torch.ops.llama.*
            return_float_values=True,
        )
        # Match Gemma's `cache_fp_type` to the surrounding model dtype so the
        # dequantized k/v output dtype matches `q` in the matmul that follows.
        self.cache_fp_type = dtype
        # Position tracker — same name & semantics as Gemma4 KVCache so the
        # existing `InitializedMutableBufferPass(["kv_cache_pos"])` picks it up.
        self.register_buffer(
            "kv_cache_pos",
            torch.arange(0, max_seq_len),
            persistent=False,
        )

    def _quantize(self, value):
        # ExecuTorch's choose_qparams_per_token_asymmetric kernel only
        # supports float32 input (op_choose_qparams.cpp:50). Cast up from
        # float16 before quantizing, then cast the quantized int8 result
        # back (int8 is dtype-agnostic, scales are already float32).
        return super()._quantize(value.float())

    def update(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        input_pos: torch.Tensor,
    ):
        # Pass `input_pos` straight through — slicing `kv_cache_pos[:seq_len]`
        # instead would specialize the symbolic seq-len dim to its example
        # value during aten→edge lowering (same bug task #20 fixed in the
        # float `KVCache`).
        #
        # Layout: parent storage is [B, S, H, D]. Parent.update expects inputs
        # as [B, H, S, D] (Llama convention) and emits two view-only
        # transposes in + two out. When `use_bshd_kv=True` the Gemma4Attention
        # caller already hands us BSHD, and wants BSHD back — pre- and
        # post-transpose so parent's transposes cancel at the stride-view
        # level (no memory copy, compiler can fuse them away).
        from ..text import Gemma4Attention  # local: avoid cycle
        if Gemma4Attention.use_bshd_kv:
            k_bhsd = k_val.transpose(1, 2)
            v_bhsd = v_val.transpose(1, 2)
            k_out_bhsd, v_out_bhsd = super().update(input_pos, k_bhsd, v_bhsd)
            return k_out_bhsd.transpose(1, 2), v_out_bhsd.transpose(1, 2)
        return super().update(input_pos, k_val, v_val)

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_cache_scales.fill_(1.0)
        self.v_cache_scales.fill_(1.0)
        if hasattr(self, "k_cache_zero_points"):
            self.k_cache_zero_points.zero_()
            self.v_cache_zero_points.zero_()
        self.kv_cache_pos.copy_(torch.arange(0, self.max_context_length))


def replace_kv_cache_with_quantized_kv_cache(module: nn.Module) -> nn.Module:
    """Walk the module tree and replace every Gemma4 `KVCache` with the
    quantized variant. Must be called AFTER `model.setup_caches(...)`.
    """
    from ..text import Gemma4Attention  # local: avoid cycle
    Gemma4KVCache = _get_gemma_kv_cache_class()
    for name, child in module.named_children():
        if isinstance(child, Gemma4KVCache):
            # Float Gemma cache storage follows `Gemma4Attention.use_bshd_kv`:
            #   BHSD: [B, H, S, D]    BSHD: [B, S, H, D].
            if Gemma4Attention.use_bshd_kv:
                B, S, H, D = child.k_cache.shape
            else:
                B, H, S, D = child.k_cache.shape
            dtype = child.k_cache.dtype
            setattr(
                module, name,
                Gemma4QuantizedKVCache(
                    batch_size=B, max_seq_len=S,
                    num_kv_heads=H, head_dim=D, dtype=dtype,
                ),
            )
        else:
            replace_kv_cache_with_quantized_kv_cache(child)
    return module
