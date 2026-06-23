---
name: xnnpack-export
description: XNNPACK backend knowledge for exporting ExecuTorch models — lowering, quantization (8da4w), tolerances, iOS static-lib gotchas.
---

# XNNPACK Export

XNNPACK is the default CPU backend for ExecuTorch on Android and iOS simulators.

## Lowering

The harness lowers via:

    to_edge_transform_and_lower(exported_program, partitioner=[XnnpackPartitioner()]).to_executorch()

`extended_ops=True` on the partitioner widens op coverage; enable it when the
default partition leaves ops on the portable fallback.

## Quantization

- **8da4w** = int8 dynamic activations + int4 weights. The standard recipe for
  text models on XNNPACK. Apply with the torchao/quantizer pass BEFORE lowering.
- Parity tolerance for quantized models is much looser than fp32 — the harness
  default is `1e-2` (vs `1e-4` for fp32). Set `--atol` if a specific model needs more.

## Parity expectations

fp32 XNNPACK matches eager to ~`1e-4`. If parity fails at fp32, suspect a wrong
wrapper (pooling/normalize mismatch) before loosening the tolerance.

## iOS

XNNPACK ships in the iOS static libs. See `dependencies/ios_static_libs.md` for
the linker setup. (MLX is physical-device-only and is a separate backend.)
