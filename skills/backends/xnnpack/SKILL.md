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
- Validate quantized output against both the transformed eager model and the
  original source, including task-level quality on held-out data. Choose
  model/output-specific tolerances before tuning; the harness has no built-in
  parity thresholds or `--atol` flag.

## Parity expectations

For fp32, `1e-4` can be a starting point for some outputs, not a universal
guarantee. Check wrapper semantics (pooling/normalization), preprocessing, and
contiguous inputs before changing thresholds. Follow
[validate-and-optimize](../../validate-and-optimize/SKILL.md) for per-output
error metrics, task quality, and target-device benchmarks.

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

## Profiling & delegation check

Start with the bundled `tools/executorch-profiler/profile_et.py` and the
[profiler guide](../../validate-and-optimize/references/profiling.md) for build
setup and commands. Supply saved representative inputs via `--model-input`, a
fresh existing `--output` directory, and `--n-tests` for repeated calls. The tool
profiles `forward` and prints per-op timing/delegation from its ETDump; it does
not validate outputs or separate warmups from measured iterations.

Investigate hot portable ops before tuning quantization or thread count.
`extended_ops=True` can widen coverage in versions that support it; check the
pinned partitioner API and revalidate outputs after changing it. More delegated
ops are useful only if target-device measurements improve. Record thread count,
warmups, median/p95 latency, and artifact hash using
[validate-and-optimize](../../validate-and-optimize/SKILL.md).
