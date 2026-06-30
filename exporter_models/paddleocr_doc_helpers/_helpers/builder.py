"""PaddleOCR doc-helpers — ONE fused 4-method .pte per backend, harness-facing.

Adapted from the standalone `export_fused_helpers.py`; the argparse / verify shell
is gone — the harness authors config.json and calls `build_and_lower(backend, out_path)`.

  orientation        [1,3,224,224]                          -> logits[1,4]
  dewarp             [1,3,712,488]                          -> grid[1,2,45,31]
  table_encode       [1,3,488,488]                          -> feat[1,256,96]
  table_decode_step  (feat[1,256,96], hidden[1,256], onehot[1,50])
                                                            -> (probs[1,50], hidden[1,256])

Backend alone fixes the lossless-maximal precision (real-image task-metric verified):
  xnnpack : CPU. DYNAMIC int8 (qd8: int8 weights, runtime-quantized activations) —
            lossless AND ~halves size vs fp32. fp16 is a no-op on XNNPACK.
  coreml  : ANE. weight-only int8 (immune to calibration drift), fp16 compute.
  vulkan  : GPU. fp16 — except the whole TABLE path (encode + decode), routed to
            XNNPACK (mixed-delegate): fp16 GPU table_encode drifts the AR decode on
            hard tables, and a tiny recurrent step is dispatch-bound on a GPU. The
            CNN methods (orientation, dewarp) stay on GPU.

The recurrent decode (`table_decode_step`) stays fp32 on every backend — quant/fp16
error compounds across the data-dependent AR loop (which itself runs in C++).
"""
import os
from torch.export import export as te

from .arch import helpers
from .wrappers import HalfWrap
from .calib import load_images
from .lowering import lower_save as _lower_save
from .backends import xnnpack, coreml, vulkan

DEFAULT_PRECISION = {"xnnpack": "int8dyn", "coreml": "int8", "vulkan": "fp16"}
DECODE = "table_decode_step"
# On Vulkan, the WHOLE table path runs on XNNPACK (mixed-delegate): encode (dyn-int8,
# lossless) + decode (fp32) go to CPU; the CNN methods stay on GPU.
VK_ON_CPU = {DECODE, "table_encode"}


def build_programs(backend, precision):
    programs = {}
    for name, mod in helpers.load_modules().items():
        ex = helpers.example_inputs(name)
        if name == DECODE:                                       # always fp32 (AR-stable)
            programs[name] = te(mod, ex, strict=False)
        elif backend == "vulkan" and name == "table_encode":     # -> XNNPACK dyn-int8 (lossless)
            hw = (ex[0].shape[2], ex[0].shape[3])
            programs[name] = xnnpack.pt2e_int8(
                mod, ex, load_images(hw, normalize=helpers.CALIB_NORM[name]), is_dynamic=True)
        elif backend == "xnnpack" and precision in ("int8", "int8dyn"):
            hw = (ex[0].shape[2], ex[0].shape[3])
            programs[name] = xnnpack.pt2e_int8(
                mod, ex, load_images(hw, normalize=helpers.CALIB_NORM[name]),
                is_dynamic=(precision == "int8dyn"))
        elif precision == "fp16" and backend != "coreml":        # fp32 I/O, half compute
            programs[name] = te(HalfWrap(mod), ex, strict=False)
        else:                                                    # fp32, or coreml fp32-graph
            programs[name] = te(mod, ex, strict=False)
    return programs


def partitioners(backend, precision, programs):
    if backend == "xnnpack":
        return [xnnpack.partitioner()]
    if backend == "vulkan":                                      # mixed-delegate per-method
        vk = vulkan.partitioner()
        return {n: ([xnnpack.partitioner()] if n in VK_ON_CPU else [vk]) for n in programs}
    if precision == "int8":
        return [coreml.partitioner("fp16", quantize="int8")]
    return [coreml.partitioner(precision)]


def build_and_lower(backend, out_path):
    """Build the fused 4-method helpers PTE for `backend`, write it to absolute `out_path`."""
    assert os.path.isabs(out_path), "out_path must be absolute"
    if backend not in DEFAULT_PRECISION:
        raise ValueError(f"unknown backend '{backend}'")
    precision = DEFAULT_PRECISION[backend]
    print(f"[build] fused helpers ({backend}/{precision}): {'+'.join(helpers.METHODS)}", flush=True)
    programs = build_programs(backend, precision)
    return _lower_save(programs, partitioners(backend, precision, programs), out_path, width=18)
