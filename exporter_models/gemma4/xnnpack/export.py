"""Gemma 4 E2B text decoder, XNNPACK backend, int4.

Validated on device. Ships as the `xnnpack/` half of the React-Native text
bundle.

Single-method PTE. The method MUST be named `forward`:
`extension/llm/runner/text_decoder_runner.h` hardcodes `load_method("forward")`
and a single-method PTE named anything else aborts at load.

The settings below are the on-device-tested values, baked. Two deviations from
the bundle that actually shipped, both deliberate and both inherited from the
recipe package:

- `ple_embedding_bits = 4`, where the shipped bundle used int2. int2 saves
  ~580 MB but flipped the multimodal first-token argmax, and that int2 PTE was
  flagged as still owing an on-device chat smoke before shipping. Set
  `PLE_EMBEDDING_BITS = 2` to reproduce the shipped bundle exactly.
- `return_last_only = False`, so the runner slices the last position itself
  after a prefill chunk. The source recipe's docstring claimed True while its
  code said False; the code is what shipped.
"""
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent

# One key per MODEL DIRECTORY, shared by that model's backends: the two copies
# of `_gemma4_edge` (here and under gemma4_omni) must never share a key, or
# `export/patches.py`'s module-global applied-set lets a wrapping patch nest
# inside itself -- the exact failure its idempotence guard exists to prevent.
_EDGE_PKG = "gemma4_edge_text"

logger = logging.getLogger(__name__)

# --- quantisation -----------------------------------------------------------
LINEAR_BITS = 4
LINEAR_GROUP_SIZE = 128
QUANTIZE_MAIN_EMBEDDINGS = True
EMBEDDING_GROUP_SIZE = 32
QUANTIZE_PLE_EMBEDDINGS = True
PLE_EMBEDDING_BITS = 4
PLE_EMBEDDING_GROUP_SIZE = 32
BASE_DTYPE = torch.float16
FP16_RETARGET = False
LOGITS_DTYPE = torch.float16

# --- sequence bounds --------------------------------------------------------
# The activation ARENA is sized by PREFILL_LEN, not by MAX_SEQ_LEN; MAX_SEQ_LEN
# only sizes the KV cache. Long generation is cheap, long prefill is not.
NUM_LAYERS = 35
NUM_KV_SHARED_LAYERS = 20
MAX_SEQ_LEN = 4048
PREFILL_LEN = 1024

# --- attention / cache flags ------------------------------------------------
# Levers B+C (shared ring state) and E (skip redundant RoPE slice) are the
# validated host-op reductions; A' is the RMSNorm rsqrt swap, which XNNPACK
# needs because it refuses a non-integer power.
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
RETURN_LAST_ONLY = False

# --- known config-vs-recipe disagreements -----------------------------------
# `e2b/xnnpack/config.json` is a VERBATIM copy of the published contract, which
# does not everywhere agree with this recipe. Keys listed here are allowed to
# differ (logged at WARNING); anything else still raises.
#
#   get_max_seq_len   config 4048, recipe 1024. The published config sets it
#                     equal to get_max_context_len, whereas the runner reads it
#                     as the PER-CALL PREFILL CHUNK SIZE (see MODEL_INTERFACE.md
#                     and dropped rule R8). Note the published config's own
#                     `forward` shape caps the seq dim at 2048, so 4048 does not
#                     agree with that either. A runner that chunks prefill at
#                     4048 will overrun this graph.
KNOWN_CONFIG_DEVIATIONS = frozenset({"get_max_seq_len"})


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
    # IMPORT ORDER IS LOAD-BEARING. `executorch.runtime.Runtime` must be
    # imported before `executorch.kernels.quantized`, or the quantized kernel
    # registration silently no-ops and `embedding_{2,4}bit.out` fails to
    # resolve much later, at load_method time, with an error that points
    # nowhere near here. Keep these three first.
    from executorch.runtime import Runtime  # noqa: F401
    import executorch.kernels.quantized  # noqa: F401
    import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401

    _load_edge()
    edge_config = importlib.import_module(f"{_EDGE_PKG}.export.config")
    harness = importlib.import_module(f"{_EDGE_PKG}.export.harness")
    pipeline = importlib.import_module(f"{_EDGE_PKG}.export.pipeline")

    variant = config["variant"]
    recipe = harness.HarnessRecipe(
        name="xnnpack-text",
        backend="xnnpack",
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
            max_seq_len=MAX_SEQ_LEN,
            prefill_len=PREFILL_LEN,
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
        # LLM runner constants come from config.json, the single source of
        # truth; `check_constants` refuses a config that contradicts the graph.
        constants=dict(variant.get("constants", {})),
    )
    harness.check_constants(recipe, KNOWN_CONFIG_DEVIATIONS)

    checkpoint = harness.resolve_checkpoint()
    out_path = Path(out_dir) / variant["file"]
    logger.info("Gemma4 E2B text / xnnpack / int4 from %s", checkpoint)
    return str(
        pipeline.export(recipe, checkpoint=Path(checkpoint), out_path=out_path)
    )
