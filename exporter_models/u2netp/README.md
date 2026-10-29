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

The exports take RGB float32 `[1,3,320,320]` input in `[0,1]` and return one
float32 `[1,1,320,320]` foreground logit map. Image-maximum scaling and ImageNet
normalization are included in the graph. RNE applies sigmoid and returns an
opaque RGBA grayscale mask; use its foreground intensity as an alpha mask when
compositing the original image. Fine structures can be missed. FP16 favors
speed and may shift mask boundaries; FP32 preserves the reference more closely.

## XNNPACK quantization

INT8 was exported and tested but is not published: foreground masks drift too
far from the FP32 reference. A mixed INT8/FP32 candidate preserved more detail
but still changed boundaries materially.

Mask IoU against the shipped FP32 `.pte`, with a probability threshold of 0.5,
on three upstream images held out from quantization calibration:

| Image       | Static INT8 | Mixed INT8/FP32 |
| ----------- | ----------- | --------------- |
| `bike.jpg`  | 0.6692      | 0.9629          |
| `boat.jpg`  | 0.8703      | 0.9410          |
| `whisk.png` | 0.9387      | 0.9945          |

These are export-agreement checks, not ground-truth segmentation accuracy.
See [XNNPACK precision notes](xnnpack/NOTES.md) for calibration, size/latency
results, the rejected FP16 trial, and profiler findings.

## Repository structure

```
config.json                                  28 B
coreml/config.json                         1.5 kB
coreml/u2netp_coreml_fp16.pte                3.3 MB
coreml/u2netp_coreml_fp32.pte                5.5 MB
xnnpack/config.json                        888 B
xnnpack/NOTES.md                           4.3 kB
xnnpack/u2netp_xnnpack_fp32.pte              4.6 MB
LICENSE                                  11.4 kB
sha256.json                                314 B
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
