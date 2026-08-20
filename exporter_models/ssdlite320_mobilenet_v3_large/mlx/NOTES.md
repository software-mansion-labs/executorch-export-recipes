# MLX is not shipped for this model

The MLX artifacts for this model were removed deliberately. The exporter in
`export-scripts` is kept so the decision can be re-tested against a future
ExecuTorch/MLX release, but no `.pte` or `config.json` is published here.

## Why

Measured on device (iPhone 16, ExecuTorch 1.4.1, Release build, 3 warmup runs +
15 timed runs, median), on ssdlite320 (mobilenet_v3_large):

| backend | precision | median latency |
|---|---|---|
| Core ML | fp32 (precision matched) | 9.2 ms |
| MLX | fp32 (only precision offered) | 17.0 ms |
| Core ML | fp16 (shipped default) | **9.7 ms** |

MLX is **1.9x slower than Core ML at matched precision**, and **1.75x slower than
the Core ML build that actually ships**.

These exporters only ever declared an MLX fp32 variant, so fp32-vs-fp32 is the
fair comparison and it is deliberately the conservative one: MLX still loses it.

**Caveat specific to this model.** The 1.75x margin against the shipped Core ML
default was the narrowest in this study. A follow-up compute-unit sweep has since
improved the Core ML side: pinning `compute_unit=CPU_ONLY` gives 7.2 ms, so the
real margin is **2.4x** (17.0 vs 7.2 ms). The removal therefore stands more
firmly than the table above suggests.

**Correction.** An earlier revision of this file stated that Core ML fp16 is
*slower* than its own fp32 for this model (9.7 vs 9.2 ms). That was measured on a
single run and did not reproduce: a repeat gave fp32 9.6 ms and fp16 9.5 ms, i.e.
the ordering flipped, and both gaps sit inside run-to-run variance of about
0.5 ms with fully overlapping min/p90 distributions. The correct statement is
that **fp16 buys no speed on this model**, not that it is slower.

The reason is that the Apple Neural Engine is a net loss here. Pinned compute
units, fp16, iPhone 16:

| compute unit | median |
|---|---|
| ALL (default) | 9.5 ms |
| CPU_AND_NE | 9.7 ms |
| CPU_AND_GPU | 11.8 ms |
| **CPU_ONLY** | **7.2 ms** |

`ALL` and `CPU_AND_NE` are indistinguishable, so the default routes to the ANE,
and dropping the ANE is 24% faster. That also explains why fp16 shows no gain:
on the ANE path, conv math precision is not the bottleneck. This is a question
about Core ML compute-unit defaults, not about MLX.

## Context

MLX was measured behind Core ML on every convolutional vision model tested. The
margin varies widely with architecture, so each model carries its own number
rather than a blanket figure.

Two things compound the gap. MLX weight quantization cannot shrink a conv model:
`quantize_model_` only rewrites `nn.Linear` and embedding modules, and
EfficientNet-V2-S holds just 6.0% of its parameters in its single `nn.Linear`,
so 4-bit quantization buys about 5% file size. Core ML by contrast reaches the
convolutions.

## Scope

This concerns convolution-dominated vision models. It does **not** generalize:

- `rfdetr-nano` (conv + transformer) measured only **1.24x** behind Core ML.
- The `lfm2.5` text encoders measured **2.7-3.1x faster on MLX** than XNNPACK.

MLX suits matmul-heavy graphs. The right backend depends on model class, so this
file is not a statement about MLX in general.

## Availability history

MLX for this model was only ever published on `main` / `v0.10.0`. No earlier tag
(`v0.9.0` and below) carried an MLX artifact for it, so there is no pinned
revision from which these files can still be fetched. They remain recoverable
from this repository's git history.
