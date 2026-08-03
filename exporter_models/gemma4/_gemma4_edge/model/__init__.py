"""Gemma4 edge model definitions.

- `text`    — the text decoder (`Gemma4EdgeForCausalLM`, `Gemma4EdgeConfig`)
- `vision`  — the vision tower (`Gemma4EdgeVisionTower`)
- `audio`   — the audio tower (`Gemma4EdgeAudioTower`)
- `custom_sdpa_ops` — registers the two `et_vk::` SDPA ops (import for side effect)
- `mel_frontend`    — standalone log-mel frontend
- `truncated`       — N-layer truncation rig for smoke tests
- `transforms`      — source transforms applied before export

Nothing here imports ExecuTorch except `transforms.rope` (which needs
`executorch.backends.vulkan.custom_ops_lib` for `et_vk::apply_rotary_emb`)
and `transforms.kv_quant` (which reuses Llama's `QuantizedKVCache`).
"""

__all__ = [
    "text",
    "vision",
    "audio",
    "custom_sdpa_ops",
    "mel_frontend",
    "truncated",
    "transforms",
]
