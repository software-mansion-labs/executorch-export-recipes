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
