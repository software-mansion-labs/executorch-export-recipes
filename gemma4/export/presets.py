"""The four export recipes.

Each corresponds to a PTE that was actually built. Provenance and trust level
are recorded on every one — read `validated_on_device` before believing a
default.

    name           source            ships as
    ------------   ---------------   ---------------------------------------
    xnnpack-text   iter210           RN text bundle, xnnpack/
    xnnpack-mm     iter211           RN multimodal bundle, xnnpack/
    vulkan-text    iter212           RN text bundle, vulkan/
    vulkan-mm      iter215           NOT the shipping PTE — see below

The shipping multimodal Vulkan PTE is byte-identical to iter214, which has a
recorded verdict of "text coherent, vision and audio emit whitespace only".
iter215 changes exactly one constant relative to iter214 —
`text_decoder_prefill_len` 128 -> 768 — which is the understood fix for that
failure, since a 128-token decoder bound cannot accept the 744 audio soft
tokens or 280 vision soft tokens the encoders emit. iter215 was never run on
device. `vulkan-mm` therefore carries the iter215 value and is marked
unvalidated. To reproduce exactly what ships today, pass
`--text-decoder-prefill-len 128`.
"""
from __future__ import annotations

import dataclasses

import torch

from .config import AttnFlags, LengthSpec, QuantSpec, Recipe

# ---------------------------------------------------------------------------
# XNNPACK
# ---------------------------------------------------------------------------

_XNNPACK_ATTN = AttnFlags(
    use_bshd_kv=True,
    use_dynamic_kv_slice=True,
    use_bool_mask=True,
    use_jit_mask=True,
    use_shared_ring_state=True,       # Levers B+C
    skip_redundant_rope_slice=True,   # Lever E
    use_broadcast_gqa=True,
    use_custom_sdpa=True,
    use_rsqrt=True,                   # Lever A'
    int8_kv_cache=True,
    ring_kv_cache=True,
    return_last_only=True,
)

XNNPACK_TEXT = Recipe(
    name="xnnpack-text",
    backend="xnnpack",
    modality="text",
    source_script="iter210_text_xnnpack_export.py",
    validated_on_device=True,
    quant=QuantSpec(
        linear_bits=4,
        linear_group_size=128,
        quantize_main_embeddings=True,
        embedding_group_size=32,
        quantize_ple_embeddings=True,
        # iter210 shipped int2 here. We default to int4 to match xnnpack-mm:
        # int2 saves ~580 MB but flipped the multimodal first-token argmax, and
        # iter210's own comment flags its int2 PTE as needing an on-device chat
        # smoke before shipping. Pass --ple-bits 2 to reproduce iter210 exactly.
        ple_embedding_bits=4,
        ple_embedding_group_size=32,
        base_dtype=torch.float16,
        fp16_retarget=False,
        logits_dtype=torch.float16,
    ),
    lengths=LengthSpec(
        num_layers=35,
        num_kv_shared_layers=20,
        max_seq_len=4048,
        # For the single-`forward` text recipes the arena bound and the example
        # input length are the same dim: seq_len in [1, 1024].
        prefill_len=1024,
    ),
    attn=dataclasses.replace(_XNNPACK_ATTN, return_last_only=False),
    notes=(
        "return_last_only=False so the runner can slice the last position "
        "itself. iter210's docstring claims True; its code says False."
    ),
)

XNNPACK_MM = Recipe(
    name="xnnpack-mm",
    backend="xnnpack",
    modality="multimodal",
    audio_backend="xnnpack",
    source_script="iter211_mm_xnnpack_export.py",
    validated_on_device=True,
    quant=QuantSpec(
        linear_bits=4,
        linear_group_size=128,
        # Bisected 2026-04-23: quantising QKVO on top of the MLPs compounded
        # drift far enough to flip the sampled token (10979 -> 9366).
        skip_linear_name_suffixes=("q_proj", "k_proj", "v_proj", "o_proj"),
        quantize_main_embeddings=True,
        embedding_group_size=32,
        quantize_ple_embeddings=True,
        ple_embedding_bits=4,
        ple_embedding_group_size=32,
        base_dtype=torch.float16,
        fp16_retarget=False,
        logits_dtype=torch.float16,
        vision_quant="8da4w",
        vision_group_size=128,
        # Audio stays fp32. An fp16 audio PTE gave min_cos 0.85 on a row that
        # flipped argmax 818 -> 100, despite eager fp16 measuring 0.99996.
        audio_quant="none",
    ),
    lengths=LengthSpec(
        num_layers=35,
        num_kv_shared_layers=20,
        max_seq_len=4048,
        prefill_len=1024,
        text_decoder_prefill_len=1024,
        vision_h=448,
        vision_w=640,
        pool_k=2,
        max_soft_tokens=280,
        audio_block_samples=7680,
        audio_k_min=1,
        audio_k_max=62,
        audio_soft_tokens_max=744,
    ),
    attn=_XNNPACK_ATTN,
)

# ---------------------------------------------------------------------------
# Vulkan
# ---------------------------------------------------------------------------

# The Vulkan SDPA source transform REJECTS ring KV, use_dynamic_kv_slice and
# use_bshd_kv=False by raising. Everything the XNNPACK recipes turn on for host
# op reduction is therefore unavailable here; the fused custom op does that work
# in the shader instead.
_VULKAN_ATTN = AttnFlags(
    use_bshd_kv=True,
    use_dynamic_kv_slice=False,
    use_bool_mask=False,
    use_jit_mask=False,
    use_shared_ring_state=False,
    skip_redundant_rope_slice=False,
    use_prebroadcast_rope=False,
    use_broadcast_gqa=False,
    use_custom_sdpa=False,
    use_rsqrt=True,
    int8_kv_cache=False,
    ring_kv_cache=False,
    vulkan_sdpa_transform=True,
    vulkan_rope_transform=True,
    return_last_only=False,
)

VULKAN_TEXT = Recipe(
    name="vulkan-text",
    backend="vulkan",
    modality="text",
    source_script="iter212_text_vulkan_export.py",
    validated_on_device=True,
    quant=QuantSpec(
        linear_bits=4,
        linear_group_size=64,
        quantize_main_embeddings=True,
        embedding_group_size=64,
        quantize_ple_embeddings=True,
        ple_embedding_bits=2,
        ple_embedding_group_size=32,
        # Loaded bf16, then retargeted: quantised stack -> fp16, scales -> fp32.
        base_dtype=torch.bfloat16,
        fp16_retarget=True,
        # iter212 leaves logits_dtype at its fp32 default while the rest of the
        # model is fp16. Preserved rather than "fixed" — changing it is a
        # numerics change that was never measured.
        logits_dtype=torch.float32,
    ),
    lengths=LengthSpec(
        num_layers=35,
        num_kv_shared_layers=20,
        max_seq_len=2048,
        prefill_len=128,
    ),
    attn=_VULKAN_ATTN,
    notes=(
        "get_max_seq_len=128 is deliberate: the forward's seq dim is bounded at "
        "128 by construction, so runners must chunk prefill at <=128. KV cache "
        "still supports 2048 positions via get_max_context_len."
    ),
)

VULKAN_MM = Recipe(
    name="vulkan-mm",
    backend="vulkan",
    modality="multimodal",
    # Audio runs on XNNPACK inside an otherwise-Vulkan PTE: the Vulkan
    # gemma_custom_sdpa per-head + softcap shader measured cos 0.40.
    audio_backend="xnnpack",
    source_script="iter215_mm_vulkan_tds768_export.py",
    validated_on_device=False,
    quant=QuantSpec(
        linear_bits=4,
        linear_group_size=128,
        quantize_main_embeddings=True,
        embedding_group_size=32,
        quantize_ple_embeddings=True,
        ple_embedding_bits=2,
        ple_embedding_group_size=32,
        base_dtype=torch.bfloat16,
        fp16_retarget=True,
        logits_dtype=torch.float16,
        # Vision stays fp32 and unquantised (the iter193b finding). This is
        # ~3.5 GB of the 4.37 GB PTE, and is the single largest open size lever.
        vision_quant="none",
        audio_quant="none",
    ),
    lengths=LengthSpec(
        num_layers=35,
        num_kv_shared_layers=20,
        max_seq_len=2048,
        # token_embedding must cover the whole multimodal id sequence.
        prefill_len=2048,
        # iter215's single change vs iter214 (which used 128 and produced
        # whitespace for vision and audio). Must exceed max(280 vision soft
        # tokens, 744 audio soft tokens) or the encoders overflow the bound.
        text_decoder_prefill_len=768,
        vision_h=448,
        vision_w=640,
        pool_k=2,
        max_soft_tokens=280,
        audio_block_samples=7680,
        audio_k_min=1,
        audio_k_max=62,
        audio_soft_tokens_max=744,
    ),
    attn=_VULKAN_ATTN,
    notes=(
        "UNVALIDATED ON DEVICE. text_decoder_prefill_len=768 is iter215's "
        "proposed fix for iter214's vision/audio whitespace failure, never run "
        "on an S24. To reproduce the PTE that ships today (and its known "
        "failure) pass --text-decoder-prefill-len 128 --force; validation rule "
        "R5 refuses that configuration by design, because it is the bug."
    ),
)


PRESETS: dict[str, Recipe] = {
    r.name: r for r in (XNNPACK_TEXT, XNNPACK_MM, VULKAN_TEXT, VULKAN_MM)
}


def get(name: str) -> Recipe:
    if name not in PRESETS:
        raise KeyError(f"unknown recipe {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
