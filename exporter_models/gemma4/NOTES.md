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

## Text runner contract (react-native-executorch TypeScript runner)

All three backends export `forward(tokens[1, S], input_pos[1]) -> float32
[1, vocab]` with an embedded `get_model_schema`, at the context length in the
vendored `e2b/<backend>/config.json` (the single source of truth for the runner
constants and the I/O contract). The configs describe what we export, not the
older published `.pte`s (vector `input_pos`, `[1, S, vocab]` logits).

| backend | port | prefill bound (`get_max_seq_len`) | notes |
|---|---|---|---|
| xnnpack | `_gemma4_edge/` | 1024 | quantized from the bf16 checkpoint, then retargeted to fp16 |
| vulkan | `_gemma4_edge/` | 128 | per-layer embeddings int4 (the 1.4.1 fork's Vulkan partitioner does not claim int2) |
| mlx | `_text_decoder/` | 512 (graph bound 511, the sliding window) | `compile_friendly=False`, non-strict trace |

- `_edge_adapters.py` fits the edge recipes to the contract by swapping names
  inside `_gemma4_edge.export.pipeline` (start position expanded in-graph via a
  cumsum, last-position fp32 logits); `_gemma4_edge/` itself is unmodified.
- `_lean_load.py` builds the model without initializing weights and assigns the
  memory-mapped checkpoint. Without it E2B (~5B params) materializes in fp32
  first and no 16 GB machine survives the load. The MLX exporter likewise
  quantizes the 2.35B-param per-layer embedding table in row chunks.
- Export with the software-mansion-labs ExecuTorch fork (`@ms/separate-backends-1.4.1`):
  the PyPI wheel serializes MLX graphs with a different op numbering, which the
  RNE runtime rejects with `InvalidProgram`.
