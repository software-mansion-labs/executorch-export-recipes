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

## Profiling & delegation check

A produced `.pte` is not verified until you confirm its ops actually run on
XNNPACK instead of falling back to the CPU portable kernels. Use the sibling
[executorch-profiler](https://gl.swmansion.com/swm-ai/executorch-profiler),
which drives an event-traced ExecuTorch runtime and reports per-op time plus a
**delegation ratio** per op.

```bash
# One-time: build ExecuTorch with the event tracer + devtools enabled.
# (EXECUTORCH_ENABLE_EVENT_TRACER=ON — a normal ET install won't emit an ETDump.)
./build_et.sh                                   # XNNPACK is always on

# Then, from the profiler dir, against your exported .pte:
python profile_et.py --model=/path/to/model.pte --output=.
```

Inputs: the profiler auto-generates random inputs from the model's static
metadata, or you pass real ones — a `generate_inputs()` you fill in, or
`--model-input inputs.pt`. `--info` just prints method metadata (shapes/dtypes)
and exits; `--n-tests N` averages over N `forward` runs.

Read the **Delegated (%)** column: GREEN `100%` = the op ran on XNNPACK, RED
`0%` = it fell back to a CPU portable kernel, YELLOW = partial. This is exactly
the signal behind the `extended_ops=True` note above — if the profiler shows ops
you expected XNNPACK to take stuck at 0%, re-lower with `extended_ops=True` (or
fix the wrapper) and re-profile. It also tells you where time actually goes,
which is the input to any quantization decision.
