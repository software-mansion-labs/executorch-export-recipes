---
name: xnnpack-export
description: XNNPACK backend knowledge for exporting ExecuTorch models — lowering (shared helper vs inline), quantization (8da4w), parity tolerances.
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

## Usage in export.py

Simple single-method XNNPACK models use the shared helper:
`from exporter.backends import lower; lower(exported_program, config["backend"])`.
Models with custom passes (e.g. rfdetr's `AddmmToLinearTransform` +
`_core_aten_ops_exception_list=[linear]`) or multiple methods lower inline with
`to_edge_transform_and_lower(..., partitioner=[XnnpackPartitioner()])`.

## iOS

XNNPACK is the default CPU backend on Android and iOS. (CoreML and MLX are
separate backends; MLX is physical-device-only — see `skills/backends/coreml`
and `skills/backends/mlx`.)
