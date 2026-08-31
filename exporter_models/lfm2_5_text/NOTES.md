# lfm2_5_text

## The XNNPACK 8da4w recipe changed

The first export of this model shipped a uniform `group_size 64` with no
embedding quantization. The sweep the old `exports/llms` directory ran (size /
throughput / perplexity / KL-divergence Pareto over fp16 and nine 8da4w
group-size combinations) picked different, per-size winners:

| size | sweep winner | what `_params.py` now sets |
|---|---|---|
| 350m | `8da4w_gs32_emb8_32` | `group_size 32`, `embedding_quantize "8,32"` |
| 1_2b | `8da4w_gs32_emb8_0` | `group_size 32`, `embedding_quantize "8,0"` |

Probing the published files confirms they predate this: neither carries a
quantized embedding table, matching the old uniform recipe rather than the
sweep. So `lfm_2_5_{350m,1_2b}_xnnpack_8da4w.pte` need a re-export to pick up
the recipe; the fp16 variants are unaffected.

The MLX backend quantizes to `4w` on its own schedule, unrelated to this sweep,
so it keeps its own `MLX_GROUP_SIZE`.

## Config

All four configs said `get_n_layers: null` where the `.pte` carry `16`; fixed.
Otherwise every published `.pte` here matches its config exactly -- the sequence
dim was already written as the real 2047 rather than 2048, unlike the six
families ported alongside it.

```bash
uv run --project exporter_models/lfm2_5_text python tools/probe_published_pte.py lfm2_5_text --size 350m
```

## It still relies on executorch's checkpoint auto-download

Unlike the six families ported alongside it, this model leaves
`cfg.base.checkpoint` unset and lets `export_llama` resolve `model_class`
through its own `HUGGING_FACE_REPO_IDS`. That happens to be safe today -- the
table points `lfm2_5_350m` / `lfm2_5_1_2b` at `LiquidAI/LFM2.5-350M` and
`LiquidAI/LFM2.5-1.2B-Instruct`, which are the repos we want -- but it means an
executorch bump could silently repoint the weights. Worth converging on the
explicit-checkpoint shape the other models use; left alone here to keep this
change to the quantization recipe.

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
