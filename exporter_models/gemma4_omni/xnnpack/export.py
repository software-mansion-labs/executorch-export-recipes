"""Gemma 4 E2B omnimodal (text + vision + audio), XNNPACK backend, int4.

Four methods in ONE .pte: `token_embedding`, `text_decoder`, `vision_encoder`,
`audio_encoder`. Every method lowers on XNNPACK with
`[XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]`, DQ FIRST --
the order is load-bearing. With the plain partitioner first it const-folds the
dequantised weights into its delegate blob and the 8da4w size saving vanishes
entirely (measured 1.00x, i.e. none).

Two quantisation decisions that are the result of bisects, not defaults:

- **q/k/v/o_proj are excluded** from linear quantisation. Bisected 2026-04-23:
  quantising QKVO on top of the MLPs compounded drift far enough to flip the
  sampled token (10979 -> 9366). They stay fp16.
- **Audio stays fp32.** An fp16 audio PTE gave min_cos 0.85 on a row that
  flipped the first-token argmax 818 -> 100, despite eager fp16 measuring
  0.99996.
"""
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent

# One key per MODEL DIRECTORY. Distinct from the gemma4 (text) model's
# `gemma4_edge_text`: the two vendored copies must not share module state, or
# `export/patches.py`'s module-global applied-set lets a wrapping patch nest
# inside itself.
_EDGE_PKG = "gemma4_edge_omni"

logger = logging.getLogger(__name__)

# --- quantisation -----------------------------------------------------------
LINEAR_BITS = 4
LINEAR_GROUP_SIZE = 128
SKIP_LINEAR_NAME_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
QUANTIZE_MAIN_EMBEDDINGS = True
EMBEDDING_GROUP_SIZE = 32
QUANTIZE_PLE_EMBEDDINGS = True
PLE_EMBEDDING_BITS = 4
PLE_EMBEDDING_GROUP_SIZE = 32
BASE_DTYPE = torch.float16
FP16_RETARGET = False
LOGITS_DTYPE = torch.float16
VISION_QUANT = "8da4w"
VISION_GROUP_SIZE = 128
AUDIO_QUANT = "none"

# --- sequence bounds --------------------------------------------------------
# PREFILL_LEN bounds token_embedding (it must cover the whole multimodal id
# sequence); TEXT_DECODER_PREFILL_LEN independently bounds text_decoder and is
# what the activation arena scales with.
NUM_LAYERS = 35
NUM_KV_SHARED_LAYERS = 20
MAX_SEQ_LEN = 4048
PREFILL_LEN = 1024
TEXT_DECODER_PREFILL_LEN = 1024

# --- vision -----------------------------------------------------------------
VISION_H = 448
VISION_W = 640
POOL_K = 2
MAX_SOFT_TOKENS = 280

# --- audio ------------------------------------------------------------------
# Waveform length must be EXACTLY AUDIO_BLOCK_SAMPLES * k for k in range: the
# mel frontend emits 48k-1 frames for 7680k samples and the wrapper pads one
# frame to the chunked-attention block boundary. Any other length breaks that.
AUDIO_BLOCK_SAMPLES = 7680
AUDIO_K_MIN = 1
AUDIO_K_MAX = 62
AUDIO_SOFT_TOKENS_MAX = 744
# The audio tower runs on XNNPACK here trivially (the whole PTE does); the
# field exists because the Vulkan sibling reroutes it.
AUDIO_BACKEND = "xnnpack"

# --- attention / cache flags ------------------------------------------------
USE_BSHD_KV = True
USE_DYNAMIC_KV_SLICE = True
USE_BOOL_MASK = True
USE_JIT_MASK = True
USE_SHARED_RING_STATE = True
SKIP_REDUNDANT_ROPE_SLICE = True
USE_BROADCAST_GQA = True
USE_CUSTOM_SDPA = True
USE_RSQRT = True
INT8_KV_CACHE = True
RING_KV_CACHE = True
RETURN_LAST_ONLY = True


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
    # IMPORT ORDER IS LOAD-BEARING. `executorch.runtime.Runtime` before
    # `executorch.kernels.quantized`, or `embedding_{2,4}bit.out` fails to
    # register and load_method fails later with an unrelated-looking error.
    from executorch.runtime import Runtime  # noqa: F401
    import executorch.kernels.quantized  # noqa: F401
    import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401

    _load_edge()
    edge_config = importlib.import_module(f"{_EDGE_PKG}.export.config")
    harness = importlib.import_module(f"{_EDGE_PKG}.export.harness")
    pipeline = importlib.import_module(f"{_EDGE_PKG}.export.pipeline")

    variant = config["variant"]
    recipe = harness.HarnessRecipe(
        name="xnnpack-mm",
        backend="xnnpack",
        modality="multimodal",
        audio_backend=AUDIO_BACKEND,
        validated_on_device=True,
        quant=edge_config.QuantSpec(
            linear_bits=LINEAR_BITS,
            linear_group_size=LINEAR_GROUP_SIZE,
            skip_linear_name_suffixes=SKIP_LINEAR_NAME_SUFFIXES,
            quantize_main_embeddings=QUANTIZE_MAIN_EMBEDDINGS,
            embedding_group_size=EMBEDDING_GROUP_SIZE,
            quantize_ple_embeddings=QUANTIZE_PLE_EMBEDDINGS,
            ple_embedding_bits=PLE_EMBEDDING_BITS,
            ple_embedding_group_size=PLE_EMBEDDING_GROUP_SIZE,
            base_dtype=BASE_DTYPE,
            fp16_retarget=FP16_RETARGET,
            logits_dtype=LOGITS_DTYPE,
            vision_quant=VISION_QUANT,
            vision_group_size=VISION_GROUP_SIZE,
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
            use_broadcast_gqa=USE_BROADCAST_GQA,
            use_custom_sdpa=USE_CUSTOM_SDPA,
            use_rsqrt=USE_RSQRT,
            int8_kv_cache=INT8_KV_CACHE,
            ring_kv_cache=RING_KV_CACHE,
            return_last_only=RETURN_LAST_ONLY,
        ),
        constants=dict(variant.get("constants", {})),
    )
    harness.check_constants(recipe)

    checkpoint = harness.resolve_checkpoint()
    out_path = Path(out_dir) / variant["file"]
    logger.info("Gemma4 E2B omni / xnnpack / int4 from %s", checkpoint)
    return str(
        pipeline.export(recipe, checkpoint=Path(checkpoint), out_path=out_path)
    )
