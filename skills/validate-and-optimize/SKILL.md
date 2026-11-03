---
name: validate-and-optimize
description: Validate exported ExecuTorch model outputs and optimize inference speed while preserving an explicit accuracy budget. Use for parity checks, quantization, delegate tuning, and target-device benchmarking in this repository.
---

# Validate and optimize an export

An artifact is ready only when its contract and output quality pass on the
intended runtime. An optimization additionally needs a reproducible improvement
on the target device. The harness does not supply automatic parity tests or a
universal `--atol` option; author model-specific validation beside the model.

## Establish the baseline and acceptance criteria

Record the upstream checkpoint revision, preprocessing/tokenizer, wrapper,
config, environment lock, exporter/runtime commits, and baseline artifact hash.
Use `eval()` and inference mode where appropriate. Fix seeds, but save and
reuse the actual inputs: seeding alone does not ensure different runtimes see
the same bytes. Reset stateful models between independent cases.

Define thresholds **before** tuning and keep two baselines distinct:

- Compare an unquantized export with the original eager source to establish
  export fidelity. A wrapper with pooling, normalization, decoding, or output
  reordering must also be checked against the original task pipeline.
- Compare reduced-precision/quantized candidates with both their transformed
  eager graph (to locate backend error) and the original source (to measure
  the total quality loss). Matching a quantized reference alone is insufficient.

"Exact" needs a per-output definition. Require exact values for deterministic
integer/bool metadata, token IDs where the test requires them, and shapes,
dtypes, output order, and method names. For floating point, choose justified
`atol`/`rtol` by output and precision; report max absolute error, relative error
with a documented near-zero floor, and how many values exceed the budget.
Check non-finite values before allclose; unexpected NaN/Inf is a failure.
Numerical thresholds are model-specific, not a blanket backend guarantee.

Add a task-level quality budget for quantization or altered precision: e.g.
WER/CER for ASR/OCR, mAP/IoU for detection/segmentation, embedding retrieval or
cosine metrics, and held-out loss/perplexity for LLMs. Keep calibration data
separate from evaluation data. A plausible image or fluent sentence is not a
quality test. Never widen a threshold just to pass a faster candidate.

## Validate the chain

Run identical fixtures through the stages that are available:

```text
original eager pipeline -> export wrapper -> captured graph
                         -> lowered program -> loaded .pte on target runtime
```

Compare the earliest stage that diverges. Some delegated programs cannot run
in the host Python process; use the matching native/device runner and mark
unavailable intermediate comparisons instead of claiming they passed.

- Inspect the **actual `.pte`** for method names, input/output shapes, dtypes,
  and constant methods. Check each component of a multi-PTE export. The helper
  `tools/probe_published_pte.py <model> --backend <b> --local <path>` can check
  `forward` metadata and runner constants; it does not execute the model or
  cover arbitrary multi-method contracts. Inspect those methods separately.
- Use real representative cases, hard cases, and min/intermediate/max supported
  shapes. Include masks, padding, empty/short inputs where supported, and
  repeat invocations. Use random inputs as smoke tests only; keep IDs in range.
- Materialize contiguous inputs at the runtime boundary, including tensors
  produced by `permute`. Verify layout and input/output buffer lifetime before
  attributing identical failures across variants to model math.
- For LLMs, compare prefill and teacher-forced decode logits using the same
  token prefixes and positions. Exercise cached vs uncached execution, cache
  reset, chunked prefill, and context/sliding-window boundaries when supported.
  Also test greedy generation and EOS/tool tokens. Autoregressive drift alone
  cannot identify the first bad operation; a small logit change can flip argmax.
- Check backend specifics in [XNNPACK](../backends/xnnpack/SKILL.md),
  [CoreML](../backends/coreml/SKILL.md), [MLX](../backends/mlx/SKILL.md), or
  [Vulkan](../backends/vulkan/SKILL.md). Host parity does not establish device
  parity: compute-unit selection, fp16 kernels, texture limits, and serialized
  delegate schemas depend on the shipping runtime.

## Measure before tuning

Measure the baseline and candidate on the same device, OS/runtime build,
thread count, backend/compute units, workload, and input shapes. Record these
settings, artifact hash, warmups, measured iterations, and thermal/power state.

Separate model loading/compilation and first invocation from steady-state
inference. Warm up until timings settle, synchronize asynchronous GPU work,
and measure enough iterations to report median and p95. Report memory and
artifact size alongside latency. Alternate or repeat baseline/candidate runs
when thermal drift or noise is comparable to the observed gain.

For LLMs report time to first token, prefill throughput, and decode tokens/s
separately, with prompt length, generated length, and cache position. Keep
tokenization and application overhead separate from model timing, or explicitly
report an end-to-end measurement. Never time only GPU dispatch.

Use the bundled [ExecuTorch profiler guide](references/profiling.md) for
XNNPACK timing/delegation (`tools/executorch-profiler/profile_et.py`) and CoreML
compute-device assignment (`tools/executorch-profiler/coreml_compute_plan.py`).
It covers source-build setup, saved inputs, trace outputs, and tool limitations.
For other backends, follow the relevant backend skill's profiling guidance.
A CPU fallback can dominate latency even
when most ops delegate. A high delegation percentage alone is not a speed
result, and host emulation is not a target-device benchmark.

## Optimize one variable at a time

Use the measured bottleneck to choose a candidate: redundant copies/layout
changes, missed delegation/fusion, shape buckets, thread count, CoreML compute
units, precision, or quantization. Retain the source checkpoint, I/O contract,
and baseline artifact. Store candidates in separate output directories.

After each meaningful change, rerun contract/parity checks and the relevant
task quality evaluation, then benchmark passing candidates. Follow the backend
skill for supported flags; do not copy partitioner flags between backends.
Eliminate failing candidates even if they are faster. Keep the baseline when
the gain is within noise or trades off quality/memory outside the agreed budget.

Record the result in the model's `NOTES.md`, with commands and a compact table:

| Candidate / artifact hash | Quality vs source / thresholds | Median / p95 | Memory / size | Device / runtime | Decision |
| --- | --- | --- | --- | --- | --- |
| Baseline | Measured values | Measured values | Measured values | Exact build | Reference |
| One change | Measured values | Measured values | Measured values | Same build | Keep/reject + reason |

State which checks were actually run. If the target device or reference data
is unavailable, retain the recipe and report that gap; do not label the model
device-verified or faster based on an export succeeding.
