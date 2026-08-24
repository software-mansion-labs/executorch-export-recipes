# smollm2

## Config is vendored, not fetched

Same two repo-wide errors as the other families: `get_n_layers: null` (real
counts are 30 / 32 / 24) and a `forward` sequence dim of 2048 where the `.pte`
declares 2047. Corrected here; everything else matched.

## The published 8da4w files for 135m and 360m are not really quantized

`smollm2_135m_xnnpack_8da4w.pte` is 0.56 GB against 0.27 GB for bf16, and 360m
is 1.36 GB against 0.72 GB. The metadata and IO contract of both are correct, so
this is not a mislabelled upload. The cause is a silent executorch behaviour:

`qmode=8da4w` with `group_size` unset falls back to **128**, and the quantizer's
filter then only takes a Linear when `in_features % group_size == 0`. At
dim 576 (135m) and 960 (960 % 128 = 64) that skips every attention projection
and the two MLP layers that read from `dim`; only the one Linear reading from
`hidden_dim` survives. The result is a nearly-unquantized fp32 model.

That is also why the old sweep ranked `8da4w_default` best for exactly these two
sizes: it was scoring a model that had barely been quantized, so it won the
perplexity/KL-divergence axes by construction.

`_params.py` therefore drops the sweep's winner for these two sizes and takes
the best *genuinely quantized* config its own `results.csv` measured:

| size | recipe | ppl | size |
|---|---|---|---|
| 135m | bf16 | 12.40 | 258 MB |
| 135m | `8da4w_default` (the sweep's pick) | 14.00 | **535 MB** |
| 135m | `gs32 + emb 8,0` (**used here**) | 16.61 | 101 MB |
| 135m | `gs64 + emb 8,0` | 19.10 | 97 MB |
| 360m | `8da4w_default` (the sweep's pick) | 9.84 | **1301 MB** |
| 360m | `gs32 + emb 8,0` (**used here**) | 11.29 | 241 MB |
| 360m | `gs64 + emb 8,0` | 12.72 | 230 MB |

Note the sweep recorded the 535 MB / 1301 MB sizes all along -- larger than the
bf16 they were meant to compress -- so the trap was visible in its own output.

`group_size 32` beats 64 by 13% / 11% perplexity for a few MB, and both divide
576 and 960. 1_7b (dim 2048, hidden 8192) was never affected and keeps the
sweep's `gs64 + emb 8,64`; there `gs32` diverges outright (ppl 4e9), which is
why the group size is chosen per size rather than repo-wide.

There is a real quality cost to quantizing these two for the first time: 135m
goes from ppl 14.00 (barely quantized, 535 MB) to 16.61 (101 MB). That is the
trade the sweep's Pareto was meant to expose; if the 135m/360m quantized
variants are not worth shipping at that quality, the honest alternative is to
drop them and ship bf16 only for those sizes.

## Published artifacts

The six shipped `.pte` all match this config's IO and runner constants:

```bash
uv run --project exporter_models/smollm2 python tools/probe_published_pte.py smollm2 --size 135m
```

But 135m and 360m `_8da4w.pte` were built with the trap above, so a re-export is
needed for the quantized variant to actually be quantized. bf16 is unaffected.

## Measured on device (Galaxy S26 Ultra, SM-S948B, Android 16)

Four builds per size, executorch's `llama_main` (arm64, ET 1.4.1 @226b471),
greedy, `--cpu_threads=4`:

| 135m | file | decode | RSS | "reverse a string" |
|---|---|---|---|---|
| bf16 (the original) | 0.270 GB | 14 tok/s | 433 MiB | correct |
| published `_8da4w` | 0.561 GB | 93 tok/s | 692 MiB | correct |
| **int8 per-channel (shipped)** | **0.166 GB** | **248 tok/s** | **316 MiB** | **correct** |
| int4 gs32 | 0.106 GB | 335 tok/s | 250 MiB | nonsense `re.sub` code |

| 360m | file | decode | RSS | "explain gravity to a six year old" |
|---|---|---|---|---|
| bf16 (the original) | 0.725 GB | 11 tok/s | 935 MiB | correct |
| published `_8da4w` | 1.364 GB | 37 tok/s | 1555 MiB | correct |
| **int8 per-channel (shipped)** | **0.412 GB** | **106 tok/s** | **649 MiB** | **correct** |
| int4 gs32 | 0.253 GB | 152 tok/s | 481 MiB | refuses: "I'm not a six-year-old..." |

int8 is the variant worth shipping at both sizes: smaller than the bf16 it sits
beside, far smaller than the file it replaces, an order of magnitude faster than
bf16, and it tracks the bf16 answers. 4-bit buys another ~40% size and speed and
loses the model.

## The published filename changes

Because the variant is genuinely int8 now, its `precision` is `8da8w` and the
file is `smollm2_{135m,360m}_xnnpack_8da8w.pte`, not `_8da4w.pte`. Whoever
republishes has to add the new file and update the consuming URL in
react-native-executorch's model registry; the old `_8da4w.pte` names describe a
build we are deliberately not producing any more. 1_7b is untouched and keeps
`smollm2_1_7b_xnnpack_8da4w.pte`.

## 1_7b was measured too, and keeps 4-bit

1_7b has the largest measured degradation of the three sizes (ppl 6.50 -> 11.99,
1.8x, against 1.3x at 135m and 1.25x at 360m), so it was exported as int8 and
compared on device against the published 4-bit build:

| 1_7b | file | decode | RSS | gravity | reverse a string | capital of France |
|---|---|---|---|---|---|---|
| published int4 | 1.018 GB | 29-45 tok/s | 1955 MiB | confused | correct | correct |
| int8 per-channel | 1.817 GB | 25-27 tok/s | 2104-2737 MiB | correct | correct | correct |

The int4 build's gravity answer is wrong ("if you throw the ball down, it will
come back up") and runs to the token cap, where int8 answers cleanly. But that
is one prompt out of three, and unlike 135m/360m the 4-bit build is not broken
on ordinary questions -- it just costs 1.8x the file and ~40% of the speed to
fix that one answer.

**So 1_7b keeps the sweep's 4-bit recipe.** The perplexity ratio flagged it, the
device did not confirm it, and the evidence is not strong enough to hand users a
model that is 800 MB larger and slower. If this is revisited, the missing
measurement is the bf16 build on the same prompts: if bf16 also fumbles the
gravity question then int4 is not the cause and there is nothing to fix. That
run was set up but not completed (the device dropped off USB mid-transfer).
