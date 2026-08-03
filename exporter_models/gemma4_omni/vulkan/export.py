"""Gemma 4 E2B omnimodal (text + vision + audio), Vulkan backend, int4.

- **audio_encoder lowers on XNNPACK inside this otherwise-Vulkan PTE.** The
  Vulkan `gemma_custom_sdpa` per-head + softcap shader measured cos 0.40 against
  eager.
- **Vision stays fp32 and unquantised** (the finding: the fp32 -> fp16
  cast of bf16-trained vision params inflated output magnitude 5x). This is
  ~3.5 GB of the 4.37 GB PTE and the single largest open size lever.
- **PLE int2** saves ~580 MB but was measured to flip the multimodal first-token
  argmax (eager PLE int2 cos 0.895).
"""
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent

# Same key as the xnnpack sibling: one vendored copy per model directory, and
# distinct from the gemma4 (text) model's `gemma4_edge_text`.
_EDGE_PKG = "gemma4_edge_omni"

logger = logging.getLogger(__name__)

# --- quantisation -----------------------------------------------------------
LINEAR_BITS = 4
LINEAR_GROUP_SIZE = 128
QUANTIZE_MAIN_EMBEDDINGS = True
EMBEDDING_GROUP_SIZE = 32
QUANTIZE_PLE_EMBEDDINGS = True
PLE_EMBEDDING_BITS = 2
PLE_EMBEDDING_GROUP_SIZE = 32
BASE_DTYPE = torch.bfloat16
FP16_RETARGET = True
LOGITS_DTYPE = torch.float16
VISION_QUANT = "none"
AUDIO_QUANT = "none"

# --- sequence bounds --------------------------------------------------------
NUM_LAYERS = 35
NUM_KV_SHARED_LAYERS = 20
MAX_SEQ_LEN = 2048
# token_embedding must cover the whole multimodal id sequence.
PREFILL_LEN = 2048
# 128 = the shipping bound, which is BELOW the 744 audio / 280 vision
# soft tokens the encoders emit. See the module docstring. 768 is the fix.
TEXT_DECODER_PREFILL_LEN = 128

# --- vision -----------------------------------------------------------------
VISION_H = 448
VISION_W = 640
POOL_K = 2
MAX_SOFT_TOKENS = 280

# --- audio ------------------------------------------------------------------
AUDIO_BLOCK_SAMPLES = 7680
AUDIO_K_MIN = 1
AUDIO_K_MAX = 62
AUDIO_SOFT_TOKENS_MAX = 744
# NOT "vulkan" -- see the module docstring.
AUDIO_BACKEND = "xnnpack"

# --- attention / cache flags ------------------------------------------------
USE_BSHD_KV = True
USE_DYNAMIC_KV_SLICE = False
USE_BOOL_MASK = False
USE_JIT_MASK = False
USE_SHARED_RING_STATE = False
SKIP_REDUNDANT_ROPE_SLICE = False
USE_PREBROADCAST_ROPE = False
USE_BROADCAST_GQA = False
USE_CUSTOM_SDPA = False
USE_RSQRT = True
INT8_KV_CACHE = False
RING_KV_CACHE = False
VULKAN_SDPA_TRANSFORM = True
VULKAN_ROPE_TRANSFORM = True
RETURN_LAST_ONLY = False

# --- known config-vs-recipe disagreements -----------------------------------
# `e2b/vulkan/config.json` is a VERBATIM copy of the published contract. Its
# sequence bounds agree with this recipe (get_max_seq_len 128 == the shipping
# text_decoder bound, get_max_context_len 2048), as do every audio_* key.
#
#   image_placeholder_id  config 262145, recipe 258880. 262145 is the UPSTREAM
#                         gemma3n id; MODEL_INTERFACE.md says the local Gemma4
#                         fork uses 258880 and "do not use those". The published
#                         config is internally inconsistent here, since its
#                         audio_placeholder_id (258881) IS the local fork value.
KNOWN_CONFIG_DEVIATIONS = frozenset({"image_placeholder_id"})


def _load_edge():
    """Path-load the vendored `_gemma4_edge` package under `_EDGE_PKG`."""
    if _EDGE_PKG in sys.modules:
        return sys.modules[_EDGE_PKG]
    root = _HERE.parent / "_gemma4_edge"
    spec = importlib.util.spec_from_file_location(
        _EDGE_PKG, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EDGE_PKG] = module
    spec.loader.exec_module(module)
    return module


def export(config: dict, out_dir: str) -> str:
    # IMPORT ORDER IS LOAD-BEARING -- see the xnnpack sibling.
    from executorch.runtime import Runtime  # noqa: F401
    import executorch.kernels.quantized  # noqa: F401
    import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401

    _load_edge()
    edge_config = importlib.import_module(f"{_EDGE_PKG}.export.config")
    harness = importlib.import_module(f"{_EDGE_PKG}.export.harness")
    pipeline = importlib.import_module(f"{_EDGE_PKG}.export.pipeline")

    variant = config["variant"]
    recipe = harness.HarnessRecipe(
        name="vulkan-mm",
        backend="vulkan",
        modality="multimodal",
        audio_backend=AUDIO_BACKEND,
        validated_on_device=False,
        quant=edge_config.QuantSpec(
            linear_bits=LINEAR_BITS,
            linear_group_size=LINEAR_GROUP_SIZE,
            quantize_main_embeddings=QUANTIZE_MAIN_EMBEDDINGS,
            embedding_group_size=EMBEDDING_GROUP_SIZE,
            quantize_ple_embeddings=QUANTIZE_PLE_EMBEDDINGS,
            ple_embedding_bits=PLE_EMBEDDING_BITS,
            ple_embedding_group_size=PLE_EMBEDDING_GROUP_SIZE,
            base_dtype=BASE_DTYPE,
            fp16_retarget=FP16_RETARGET,
            logits_dtype=LOGITS_DTYPE,
            vision_quant=VISION_QUANT,
            audio_quant=AUDIO_QUANT,
        ),
        lengths=edge_config.LengthSpec(
            num_layers=NUM_LAYERS,
            num_kv_shared_layers=NUM_KV_SHARED_LAYERS,
            max_seq_len=MAX_SEQ_LEN,
            prefill_len=PREFILL_LEN,
            text_decoder_prefill_len=TEXT_DECODER_PREFILL_LEN,
            vision_h=VISION_H,
            vision_w=VISION_W,
            pool_k=POOL_K,
            max_soft_tokens=MAX_SOFT_TOKENS,
            audio_block_samples=AUDIO_BLOCK_SAMPLES,
            audio_k_min=AUDIO_K_MIN,
            audio_k_max=AUDIO_K_MAX,
            audio_soft_tokens_max=AUDIO_SOFT_TOKENS_MAX,
        ),
        attn=edge_config.AttnFlags(
            use_bshd_kv=USE_BSHD_KV,
            use_dynamic_kv_slice=USE_DYNAMIC_KV_SLICE,
            use_bool_mask=USE_BOOL_MASK,
            use_jit_mask=USE_JIT_MASK,
            use_shared_ring_state=USE_SHARED_RING_STATE,
            skip_redundant_rope_slice=SKIP_REDUNDANT_ROPE_SLICE,
            use_prebroadcast_rope=USE_PREBROADCAST_ROPE,
            use_broadcast_gqa=USE_BROADCAST_GQA,
            use_custom_sdpa=USE_CUSTOM_SDPA,
            use_rsqrt=USE_RSQRT,
            int8_kv_cache=INT8_KV_CACHE,
            ring_kv_cache=RING_KV_CACHE,
            vulkan_sdpa_transform=VULKAN_SDPA_TRANSFORM,
            vulkan_rope_transform=VULKAN_ROPE_TRANSFORM,
            return_last_only=RETURN_LAST_ONLY,
        ),
        constants=dict(variant.get("constants", {})),
    )
    harness.check_constants(recipe, KNOWN_CONFIG_DEVIATIONS)

    soft_tokens = max(MAX_SOFT_TOKENS, AUDIO_SOFT_TOKENS_MAX)
    if TEXT_DECODER_PREFILL_LEN < soft_tokens:
        logger.info(
            "text_decoder seq bound is %d while the encoders emit up to %d soft "
            "tokens (vision %d, audio %d). This is BY DESIGN -- 128 is the "
            "largest seq_len one execute() may carry before the Vulkan "
            "attention cliff (S>=129). The runner MUST chunk the prefill at "
            "<=%d: llama_main_mm does (--prefill_chunk_size, default 128); "
            "stock multimodal_prefiller.cpp does NOT and will overrun. See "
            "this file's docstring.",
            TEXT_DECODER_PREFILL_LEN, soft_tokens, MAX_SOFT_TOKENS,
            AUDIO_SOFT_TOKENS_MAX, TEXT_DECODER_PREFILL_LEN,
        )

    checkpoint = harness.resolve_checkpoint()
    out_path = Path(out_dir) / variant["file"]
    logger.info("Gemma4 E2B omni / vulkan / int4 from %s", checkpoint)
    return str(
        pipeline.export(recipe, checkpoint=Path(checkpoint), out_path=out_path)
    )
