# MLX is not shipped for this model

The MLX artifacts for this model were removed deliberately. The exporter in
`export-scripts` is kept so the decision can be re-tested against a future
ExecuTorch/MLX release, but no `.pte` or `config.json` is published here.

## Why

Measured on device (iPhone 16, ExecuTorch 1.4.1, Release build, 3 warmup runs +
15 timed runs, median):

| backend | precision | median latency |
|---|---|---|
| Core ML | fp16 | **1.2 ms** |
| MLX | fp32 | 12.0 ms |

MLX is **10.0x slower than Core ML** for YOLO26-pose-n @384.

Note the precisions differ: these are each backend's *declared default* for this model. MLX fp32 and fp16 measured within 2 ms of each other on efficientnet-v2-s (45.0 vs 47.0 ms), so precision does not account for the gap.

This is not specific to this model. MLX was measured behind Core ML on every
convolutional vision model tested, by 7.5x to 19.6x:

| model | Core ML | MLX | MLX penalty |
|---|---|---|---|
| efficientnet-v2-s @384 | 2.4 ms | 47.0 ms | 19.6x |
| yolo26-s @640 | 3.7 ms | 59.2 ms | 16.0x |
| yolo26-seg-s @640 | 11.0 ms | 82.9 ms | 7.5x |
| yolo26-pose-n @384 | 1.2 ms | 12.0 ms | 10.0x |

Two things compound the latency result:

1. **MLX quantization cannot shrink a conv model.**
   `executorch.backends.mlx.llm.quantization.quantize_model_` only rewrites
   `nn.Linear` and embedding modules. EfficientNet-V2-S holds just 6.0% of its
   parameters (1,281,000 of 21,458,488) in its single `nn.Linear`, so 4-bit
   weight quantization buys about 5% file size. `nvfp4` is a literal no-op: it
   logs `Skipping classifier.1: weight shape [1000,1280] incompatible with
   nvfp4 (group_size=16)` and emits a byte-identical artifact.
2. **Core ML quantizes the whole model.** Core ML's
   `op_linear_quantizer_config` reaches the convolutions, so int8/per_channel
   halves the artifact again (43.9 MB fp16 -> 22.1 MB) at identical latency.

## Scope

This applies to convolution-dominated vision models. It does **not** generalize:

- `rfdetr-nano` (conv + transformer) measured only **1.24x** behind Core ML.
- The `lfm2.5` text encoders measured **2.7-3.1x faster on MLX** than XNNPACK.

MLX suits matmul-heavy graphs. The right backend depends on model class, so
this file is not a statement about MLX in general.

## Availability history

MLX for this model was only ever published on `main` / `v0.10.0`. No earlier tag
(`v0.9.0` and below) ever carried an MLX artifact for it, so there is no pinned
revision from which these files can still be fetched. They remain recoverable
from this repository's git history.

A commit description in the removal commit stated that earlier tags "still
contain the MLX artifacts". That is incorrect, and this section is the
correction; HuggingFace commit descriptions cannot be edited after the fact.
