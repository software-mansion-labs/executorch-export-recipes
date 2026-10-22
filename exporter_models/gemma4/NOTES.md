# gemma4 notes

Gemma 4 E2B text decoder, MLX backend, int4. Model code vendored under
`_text_decoder/` (bit-for-bit copy of the executorch
`examples/models/gemma4/text_decoder/` package); `export.py` transcribes the body
of `export_gemma4_e2b_mlx`.

## Export settings are the on-device-tested invocation (not function defaults)

The recipe is the executorch `export_mlx_standalone.py`, whose `export()` calls
`export_gemma4_e2b_mlx` with the BAKED-IN, on-device-tested settings (its docstring:
"int4 / group_size 64, bf16, compile-friendly"):

- `dtype="bf16"`, `qlinear="4w"`, `qembedding="4w"`, `group_size=64`,
  `compile_friendly=True`, `max_seq_len=2048`.

These are NOT the `export_gemma4_e2b_mlx` function-signature defaults (fp32 / gs32 /
compile_friendly=False) — they are the values the standalone passes, which produced
the published `gemma4_e2b_mlx_int4.pte`. `export.py` bakes these same values, so it is
faithful to the recipe's actual invocation.

Structural-only: the MLX backend needs an executorch source build, so the real export
runs there (the published .pte already exists on HF).

## Core ML: not supported, and not attempted

This model has **no Core ML export and is not expected to get one.** Apple-silicon
acceleration here is MLX only (see `mlx/export.py`), which additionally requires the
software-mansion-labs ExecuTorch fork -- the PyPI wheel produces an MLX payload the
RNE runtime cannot load.

**Why Core ML is not on the table.** This is an autoregressive decoder driven by a
mutable KV cache. ExecuTorch's Core ML backend has no practical path for that: the
cache is a mutable buffer carried across calls, and the decode step needs a growing
sequence dim. Core ML's dynamic-shape support is the exact area where we already
have a confirmed, filed defect -- non-CPU compute paths reject shapes Core ML infers
as data-dependent (apple/coremltools#2825, see `docs/coremltools-2825-repro.md`).

**Status: a judgment call, not a measured result.** Nobody has run this model through
the Core ML backend and collected a failure. It is not attempted because the cost is
high and the prior is bad, not because it was tried and rejected. If Core ML gains
usable stateful-KV-cache support, this is worth revisiting.
