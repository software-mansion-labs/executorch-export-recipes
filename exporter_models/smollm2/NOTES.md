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

`_params.py` therefore uses `group_size 64` for 135m and 360m instead of the
sweep's winner -- 576 and 960 are both divisible by 64, so every Linear
quantizes. 1_7b (dim 2048, hidden 8192) was never affected and keeps the sweep's
`gs64 + emb 8,64`.

## Published artifacts

The six shipped `.pte` all match this config's IO and runner constants:

```bash
uv run --project exporter_models/smollm2 python tools/probe_published_pte.py smollm2 --size 135m
```

But 135m and 360m `_8da4w.pte` were built with the trap above, so a re-export is
needed for the quantized variant to actually be quantized. bf16 is unaffected.
