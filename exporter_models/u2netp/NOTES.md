# U²-Netp background removal investigation

The exported model fits RNE's existing single-channel semantic segmentation
pipeline. No task or native changes are needed. See README.md for the published variants and compatibility. Export verification
and reproduction details are recorded below.

Published repository: https://huggingface.co/software-mansion/U2NetP-ExecuTorch
Revision tag: `v1.0.0`. The library exposes the five artifacts under
`models.semanticSegmentation.U2NETP`.

## Export changes

- Keep U2NETP and the shared RSU blocks from the pinned upstream implementation.
- Return the fused `d0` logits before sigmoid. Auxiliary prediction heads are
  not returned, and their sigmoid operations are eliminated from the export.
  Side convolutions remain because the fused prediction uses all six maps.
- Replace deprecated `F.upsample` with equivalent bilinear `F.interpolate`,
  explicitly setting `align_corners=False`.
- Include upstream image-maximum and ImageNet normalization inside the graph;
  guard the all-black input with a small positive denominator.
- Keep upstream demo output min/max contrast stretching out of the graph so
  RNE's sigmoid produces foreground probabilities.

XNNPACK has one portable `amax` input reduction and one delegated network.
CoreML has one delegate and no portable kernels. The CoreML compute plan puts
369/370 dispatched FP16 operations on the host ANE, with one cast on CPU.
The installed wheel lacks event tracing. A local event-traced ExecuTorch 1.4.1
build was subsequently used with the cloned profiler: 99.68% of execution time
was in the XNNPACK delegate and 0.31% in `amax`. See
[xnnpack/NOTES.md](xnnpack/NOTES.md) for rejected quantization candidates and
actual exported-runtime quality results.

## Validation and performance

Host checks used an Apple M1 Pro running macOS 26.5.2. These are warm host
measurements, not phone benchmarks, and exclude image preprocessing and mask
rendering in React Native.

XNNPACK runs through the ExecuTorch runtime. Its probability error against
upstream eager PyTorch is below 0.00015 on the tested bicycle, boat and animal
images. The graph contains one XNNPACK delegate plus a portable `amax` input
reduction. A later event-traced local build measured 99.68% of execution time inside the
XNNPACK delegate; see the precision notes for the profiler findings.

The actual CoreML partitions were extracted from the saved `.pte` files and
executed directly through CoreML, since the installed ExecuTorch wheel cannot
execute its multifunction CoreML packages. Both CoreML exports contain one
delegate with no portable operations. The profiler's compute-plan tool places
369 of 370 dispatched FP16 operations on the Neural Engine; one cast runs on
CPU.

On the three test images, FP16 foreground-probability MAE is 0.00089–0.00453
and thresholded-mask IoU against FP32 is 0.9873–0.9991. Individual edge pixels
can differ by up to 0.312, so FP16 is a speed/quality tradeoff. CoreML FP32
matches the eager thresholded masks on all three images, with maximum
probability error below 0.00017. Warm direct-CoreML medians are approximately
7.2 ms (FP16) and 16.7 ms (FP32).

Black/white inputs were also checked for finite results. This is a conversion
smoke test, not a dataset accuracy evaluation. Full CoreML execution through
RNE and both backends' performance require validation on the target devices.

## Reproduce the exports

From the export-scripts repository:

```sh
uv run --project exporter_models/u2netp exporter export u2netp --backend xnnpack --output-dir exports/u2netp/xnnpack
uv run --project exporter_models/u2netp exporter export u2netp --backend coreml --variant fp16 --output-dir exports/u2netp/coreml
uv run --project exporter_models/u2netp exporter export u2netp --backend coreml --variant fp32 --output-dir exports/u2netp/coreml
```

The model code is vendored from upstream commit
`ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`. The checkpoint is downloaded from the
author's linked Google Drive artifact and checked against SHA-256
`e7567cde013fb64813973ce6e1ecc25a80c05c3ca7adbc5a54f3c3d90991b854`.

## Selective XNNPACK INT8

The additional `int8` variant uses `_quantization.py` to quantize four expensive
convolutions while retaining sensitive operations in FP32. The exporter fetches
13 calibration images at the pinned upstream revision into `datasets/u2netp/`.
See `xnnpack/NOTES.md` for the operator-filter workaround, exported-runtime
validation on 13 held-out images, and interleaved host timing.

## Vulkan

The Android Vulkan FP16 export was executed on the connected Galaxy
S20+ using RNE's shipped 1.4.1 native libraries. FP16 preserves input
normalization and one sensitive convolution in FP32; entirely FP16 was rejected.
See `vulkan/NOTES.md` for precision choices, validation, phone timings and the
reproducible AOT environment. Vulkan was slower than XNNPACK on this phone, so
the Android default remains XNNPACK FP32.
