---
license: apache-2.0
pipeline_tag: image-segmentation
library_name: executorch
---

# u2netp

This repository hosts the **U²-Netp** models exported for the
[React Native ExecuTorch](https://www.npmjs.com/package/react-native-executorch)
library as ExecuTorch `.pte` programs, ready to run on device.

U²-Netp predicts a soft mask for salient foreground objects, including animals
and products as well as people. It does not classify object categories or
separate individual instances.

Upstream model: [U²-Net](https://github.com/xuebinqin/U-2-Net),
revision `ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`.

## Variants

| Path                              | Backend | Precision |
| --------------------------------- | ------- | --------- |
| `coreml/u2netp_coreml_fp16.pte`   | coreml  | fp16      |
| `coreml/u2netp_coreml_fp32.pte`   | coreml  | fp32      |
| `xnnpack/u2netp_xnnpack_fp32.pte` | xnnpack | fp32      |
| `xnnpack/u2netp_xnnpack_int8.pte` | xnnpack | selective int8 / fp32 |
| `vulkan/u2netp_vulkan_fp16.pte` | vulkan | fp16 with one fp32 convolution |

The exports take RGB float32 `[1,3,320,320]` input in `[0,1]` and return one
float32 `[1,1,320,320]` foreground logit map. Image-maximum scaling and ImageNet
normalization are included in the graph. RNE applies sigmoid and returns an
opaque RGBA grayscale mask; use its foreground intensity as an alpha mask when
compositing the original image. Fine structures can be missed. FP16 favors
speed and may shift mask boundaries; FP32 preserves the reference more closely.

## Precision notes

The INT8 variant selectively quantizes four expensive convolutions with static
INT8 activations and per-channel INT8 weights. Sensitive layers and skip
connections remain FP32; public input and output types are unchanged.

On an Apple M1 Pro, interleaved ExecuTorch runs measured **87 ms** median versus
**156 ms** for FP32 (1.78× faster; 30 runs after five warm-ups). Device speedups
will vary. Across 13 images outside calibration, exported masks averaged
**0.9953 IoU** against FP32 (minimum 0.9868), at a foreground threshold of 0.5.
These measure export agreement, not accuracy against ground-truth masks.
Choose FP32 when boundary precision matters.

Full-network INT8 remains unpublished because it loses substantial foreground
detail. See [XNNPACK precision notes](xnnpack/NOTES.md) for the selective recipe,
calibration, validation and rejected alternatives.

Vulkan FP16 preserves input normalization and one sensitive convolution in FP32.
On a Galaxy S20+ (Android 13, Mali), native forward medians were 337 ms for this
variant, versus 316 ms for XNNPACK FP32 and 189 ms for selective INT8. Vulkan is optional; Android retains its XNNPACK FP32
default. Across 13 device-tested images, Vulkan FP16 averaged 0.9945 mask IoU
against FP32 (minimum 0.9854). See [Vulkan precision notes](vulkan/NOTES.md)
for the export recipe, phone validation and timing limitations.

## Repository structure

```
config.json                                    28 B
coreml/config.json                           1.5 kB
coreml/u2netp_coreml_fp16.pte                3.3 MB
coreml/u2netp_coreml_fp32.pte                5.5 MB
vulkan/config.json                            886 B
vulkan/NOTES.md                              4.1 kB
vulkan/u2netp_vulkan_fp16.pte                2.4 MB
xnnpack/config.json                          1.5 kB
xnnpack/NOTES.md                             6.6 kB
xnnpack/u2netp_xnnpack_fp32.pte              4.6 MB
xnnpack/u2netp_xnnpack_int8.pte              4.0 MB
LICENSE                                     11.4 kB
```

## Compatibility

These files are published for the **ExecuTorch v1.4.1** runtime. ExecuTorch
gives no forward compatibility guarantee, so an older runtime may fail to load
them. CoreML requires iOS 18 / macOS 15 or newer; full RNE execution and
performance still require validation on the target device.

To use them in React Native ExecuTorch, pass
`models.semanticSegmentation.U2NETP.DEFAULT` to `useSemanticSegmenter` or
`createSemanticSegmenter`. See the
[documentation](https://docs.swmansion.com/react-native-executorch/docs/fundamentals/downloading-models).

To load these files in your own ExecuTorch runtime, read the
[compatibility note](https://github.com/pytorch/executorch/blob/main/runtime/COMPATIBILITY.md)
first.
