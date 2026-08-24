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

Re-exported 135m 8da4w with the recipe above and compared against the published
file with executorch's `llama_main` (arm64, ET 1.4.1 @226b471, greedy,
`--cpu_threads=4`):

| size | | published `_8da4w.pte` | re-exported (gs32 + emb 8,0) |
|---|---|---|---|
| 135m | file | 0.561 GB | 0.106 GB |
| 135m | RSS | 692 MiB | 250 MiB |
| 135m | decode | 81 tok/s | 223 tok/s |
| 360m | file | 1.364 GB | 0.253 GB |
| 360m | RSS | 1223 MiB | 481 MiB |
| 360m | decode | 32 tok/s | 100 tok/s |

Both are ~5.3x smaller, ~2.5x lighter in RAM and ~3x faster to decode, which is
what quantizing a model that was never actually quantized should look like.

**The quality cost is real and visible.** On "explain gravity to a six year old",
135m answers well either way, but 360m regresses outright: the published (barely
quantized) file gives a clean answer, while the re-export derails into "I'm sorry,
I'm not a six-year-old..." and runs to the token cap. One prompt is not a
measurement, but it is consistent with the ppl the sweep recorded (9.84 -> 11.29
at 360m, 14.00 -> 16.61 at 135m).

So the choice is explicit: ship a genuinely quantized 135m/360m that is 5x
smaller and 3x faster but measurably weaker, or drop the quantized variant for
these two sizes and ship bf16 only. The status quo -- a "quantized" file twice
the size of bf16 -- is the one option that makes no sense.
