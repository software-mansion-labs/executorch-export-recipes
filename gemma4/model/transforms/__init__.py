"""Source transforms applied to the model before export.

Ordering is load-bearing. The full sequence, with the constraints each step
imposes:

    model = load_from_safetensors(...)          # or load_truncated_from_safetensors
    model.model.split_ple_for_export()          # AFTER load, BEFORE quantize
    model = quantize_model(model, ...)          # AFTER split_ple_for_export
    model.setup_caches(...)                     # AFTER load (needs device)

    # XNNPACK only — both are rejected by the Vulkan SDPA transform:
    kv_quant.replace_kv_cache_with_quantized_kv_cache(model)   # AFTER setup_caches
    kv_ring.replace_sliding_kv_cache_with_ring_kv_cache(model) # AFTER kv_quant

    # Vulkan only:
    sdpa.replace_gemma_sdpa_with_custom_op(model)   # AFTER setup_caches
    rope.replace_gemma_rope_with_paired_fused(model)  # AFTER sdpa

`sdpa.replace_gemma_sdpa_with_custom_op` raises rather than degrades when it
meets a configuration it cannot handle: ring KV caches, `use_dynamic_kv_slice`,
a model whose `setup_caches` has not been called, or `use_bshd_kv=False`.
`gemma4.export.validate` catches these combinations before any work is done.
"""

__all__ = ["sdpa", "rope", "kv_quant", "kv_ring"]
