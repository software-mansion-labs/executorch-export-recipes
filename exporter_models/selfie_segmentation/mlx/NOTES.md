# MLX is not shipped for this model

The MLX artifacts for this model were removed deliberately. The exporter in
`export-scripts` is kept so the decision can be re-tested against a future
ExecuTorch/MLX release, but no `.pte` or `config.json` is published here.

## Why

Measured on device (iPhone 16, ExecuTorch 1.4.1, Release build, 3 warmup runs +
15 timed runs, median), on selfie-segmentation:

| backend | precision | median latency |
|---|---|---|
| Core ML | fp32 (precision matched) | 4.5 ms |
| MLX | fp32 (only precision offered) | 5.1 ms |
| Core ML | fp16 (shipped default) | **0.3 ms** |

MLX is **1.1x slower than Core ML at matched precision**, and **17.0x slower than
the Core ML build that actually ships**.

These exporters only ever declared an MLX fp32 variant, so fp32-vs-fp32 is the
fair comparison and it is deliberately the conservative one: MLX still loses it.

**Caveat specific to this model.** At matched precision MLX is only 1.1x behind
Core ML, which is near parity and the closest result in this study. MLX is not
performing badly here. It is removed because Core ML fp16 is exceptional on this
model (0.3 ms, a 15x gain over its own fp32), and fp16 is what ships. If Core ML
fp16 ever regressed for this model, MLX would be a reasonable fallback.

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
