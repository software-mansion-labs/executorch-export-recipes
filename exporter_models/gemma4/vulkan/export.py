"""Gemma 4 E2B text decoder, Vulkan backend, int4.

Validated on device. Ships as the `vulkan/` half of the React-Native text
bundle.

Single-method PTE named `forward` (see the xnnpack sibling for why the name is
not optional).

REQUIRES PATCHED VULKAN KERNELS, from:

    software-mansion-labs/executorch
    branch @ms/separate-backends          (ET 1.3.1 -- the SAME branch mlx uses)
    verified at 2daed17d827b7d7cc2f61d44dc20794c81d8da94

They are NOT vendored here: the softmax max-subtraction fix, the gelu/tanh fp16
clamp, `GemmaSDPA.cpp` implementing `et_vk::gemma_sdpa_with_kv_cache` and
`et_vk::gemma_custom_sdpa`, the `op_registry.py` entries the partitioner needs,
and several `resize_fn` fixes for dynamic shapes. Export and runtime must come
from the same tree state -- a PTE built against one set of shaders and run
against another produces gibberish at runtime, not an error, and that is the
first thing to check when on-device output is nonsense.

The kernels originated on the fork's `@mkopcins/gemma4` branch (ET 1.2.0a0) and
were ADAPTED onto 1.3.1, not copied: the tile loaders gained a `D` argument and
the shaders now declare `Q_LAYOUT/K_LAYOUT DHSB`. The on-device validation this
recipe claims was measured against the 1.2 shaders -- re-run the smoke test
before trusting those numbers. See NOTES.md, "Backend prerequisites".

Two settings that look like bugs and are not:

- `LOGITS_DTYPE = fp32` while the rest of the model is fp16. That is what the
  on-device-tested recipe does; it is preserved rather than "fixed", because
  changing it is a numerics change nobody has measured.
- `PREFILL_LEN = 128`, so `get_max_seq_len` is 128. The forward's seq dim is
  bounded at 128 by construction, so runners must chunk prefill at <=128. The
  KV cache still serves 2048 positions via `get_max_context_len`.

Everything the XNNPACK recipe turns on for host-op reduction is unavailable
here: the Vulkan SDPA source transform REJECTS ring KV, `use_dynamic_kv_slice`
and `use_bshd_kv=False` by raising. The fused custom op does that work in the
shader instead.
"""
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent

# Same key as the xnnpack sibling: one vendored copy per model directory.
_EDGE_PKG = "gemma4_edge_text"

logger = logging.getLogger(__name__)

# --- quantisation -----------------------------------------------------------
# Loaded bf16, then retargeted: the quantised stack becomes fp16 (what the
# Vulkan `linear_q4gsw_*` shaders want) while the per-group scales are upcast
# to fp32 (lossless -- the bf16 mantissa fits exactly). bf16 -> fp16 rounding of
# those scales was the dominant fp16 accuracy floor.
LINEAR_BITS = 4
LINEAR_GROUP_SIZE = 64
QUANTIZE_MAIN_EMBEDDINGS = True
EMBEDDING_GROUP_SIZE = 64
QUANTIZE_PLE_EMBEDDINGS = True
# int4, not the 1.3.1 recipe's int2: on the ET 1.4.1 fork the Vulkan
# partitioner claims the 4-bit embedding dequantize but not the 2-bit one, and
# the 35 per-layer `torchao::dequantize_affine` left on the host have no
# ExecuTorch kernel ("Missing out variants"). Costs ~0.6 GB of .pte size.
PLE_EMBEDDING_BITS = 4
PLE_EMBEDDING_GROUP_SIZE = 32
BASE_DTYPE = torch.bfloat16
FP16_RETARGET = True
LOGITS_DTYPE = torch.float32

# --- sequence bounds --------------------------------------------------------
NUM_LAYERS = 35
NUM_KV_SHARED_LAYERS = 20
MAX_SEQ_LEN = 2048
PREFILL_LEN = 128

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
# Slicing hidden states before lm_head produces [1, 1, vocab] which fits in
# texture and avoids broken select_copy.int shader on large vocab buffers in Vulkan.
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


def _edge_adapters():
    spec = importlib.util.spec_from_file_location("gemma4_edge_adapters", _HERE.parent / "_edge_adapters.py")
    mod = importlib.util.module_from_spec(spec)
    # Registered by name: torch.export resolves the adapters' globals by import.
    sys.modules["gemma4_edge_adapters"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    # IMPORT ORDER IS LOAD-BEARING -- see the xnnpack sibling.
    from executorch.runtime import Runtime  # noqa: F401
    import executorch.kernels.quantized  # noqa: F401
    import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401

    from exporter.schema import load_and_validate_schema

    _load_edge()
    edge_config = importlib.import_module(f"{_EDGE_PKG}.export.config")
    harness = importlib.import_module(f"{_EDGE_PKG}.export.harness")
    pipeline = importlib.import_module(f"{_EDGE_PKG}.export.pipeline")
    adapters = _edge_adapters()
    adapters.adapt(pipeline)
    adapters.vulkan_post_fusion_dtype_promotion()

    variant = config["variant"]
    # The KV cache is sized by the config's context length; the chunk size stays
    # the graph's PREFILL_LEN.
    max_seq_len = variant.get("constants", {}).get("get_max_context_len", MAX_SEQ_LEN)
    recipe = harness.HarnessRecipe(
        name="vulkan-text",
        backend="vulkan",
        modality="text",
        validated_on_device=True,
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
        ),
        lengths=edge_config.LengthSpec(
            num_layers=NUM_LAYERS,
            num_kv_shared_layers=NUM_KV_SHARED_LAYERS,
            max_seq_len=max_seq_len,
            prefill_len=PREFILL_LEN,
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
    harness.check_constants(recipe)

    # The runtime I/O spec, baked into the .pte as `get_model_schema`. The
    # rne-rewrite runtime parses it at load to learn the dynamic-dim domains
    # and the tokens/input_pos equality constraint; without it every dim
    # collapses to the compiled upper bound and a decode call at S=1 is
    # rejected. Bounds are templated because they are per-backend.
    schema_json = load_and_validate_schema(
        _HERE.parent / "schema.json",
        replacements={
            "seq_max": str(PREFILL_LEN),
            "vocab_size": str(variant["constants"]["get_vocab_size"]),
        },
    )

    checkpoint = harness.resolve_checkpoint()
    out_path = Path(out_dir) / variant["file"]
    logger.info("Gemma4 E2B text / vulkan / int4 from %s", checkpoint)
    return str(
        pipeline.export(
            recipe,
            checkpoint=Path(checkpoint),
            out_path=out_path,
            schema_json=schema_json,
        )
    )
