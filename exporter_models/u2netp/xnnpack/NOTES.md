# XNNPACK precision experiments

Only FP32 is published. Static INT8, mixed INT8/FP32, and FP16 candidates were
exported and executed, but rejected because they changed the foreground masks.
The absence of an INT8 file is a quality decision, not an export failure.

## Static INT8

The experiment used PT2E `prepare_pt2e` / `convert_pt2e`,
`XNNPACKQuantizer`, and `get_symmetric_quantization_config(is_per_channel=True)`:
static INT8 activations with per-channel INT8 weights. Public inputs and
outputs remained float32. The export included the same image-maximum and
ImageNet normalization as FP32, and returned foreground logits before sigmoid.

Calibration used 13 RGB images from the upstream repository at commit
`ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`, resized to 320×320:
`0002-01.jpg`, `0003.jpg`, `girl.png`, `hockey.png`, `horse.jpg`, `im_01.png`,
`im_14.png`, `im_21.png`, `im_27.png`, `lamp2_meitu_1.jpg`, `long.jpg`,
`rifle1.jpg`, and `sailboat3.jpg`.

The three images below were held out from calibration. Results compare actual
ExecuTorch `.pte` execution against the shipped FP32 `.pte`. Mask IoU uses a
foreground-probability threshold of 0.5; MAE is measured after sigmoid in [0,1].
These measure conversion agreement, not accuracy against ground-truth masks.

| Image       | Static INT8 mask IoU | Static INT8 probability MAE | Mixed INT8 mask IoU | Mixed INT8 probability MAE |
| ----------- | -------------------- | --------------------------- | ------------------- | -------------------------- |
| `bike.jpg`  | 0.6692               | 0.09813                     | 0.9629              | 0.01421                    |
| `boat.jpg`  | 0.8703               | 0.04655                     | 0.9410              | 0.02156                    |
| `whisk.png` | 0.9387               | 0.06170                     | 0.9945              | 0.00458                    |

Static INT8 is smaller and faster, but loses substantial foreground detail.
Those mask changes outweigh the size and latency improvements.

## Mixed INT8/FP32

The second candidate kept `stage1`, `stage2`, `stage1d`, `stage2d`, `side1`,
`side2`, and `outconv` in FP32, quantizing the remaining network with the same
calibration data. Preserving these early and final high-resolution blocks
improved agreement, but the boat mask still fell to 0.9410 IoU and the latency
gain was small. This candidate is also not published.

## Host size and latency

Interleaved measurements used the same bicycle input, three warm-up calls,
then 20 executions per candidate through ExecuTorch on an Apple M1 Pro running
macOS 26.5.2, with torch 2.14.0, ExecuTorch 1.4.1 and torchao 0.18.0.

| Candidate                  | File size | Median inference |
| -------------------------- | --------- | ---------------- |
| FP32 (published)           | 4.61 MB   | 217 ms           |
| Static INT8 (rejected)     | 1.36 MB   | 64 ms            |
| Mixed INT8/FP32 (rejected) | 2.61 MB   | 215 ms           |

Other export/build work was running on the host, so these are indicative
comparisons rather than isolated device benchmarks. They exclude RNE image
preprocessing and rendering. The small calibration/validation sample does not
prove that every quantization strategy is unsuitable; it establishes why these
particular candidates were not shipped. A future attempt needs more calibration
coverage and a larger independent quality evaluation before publication.

## FP16 trial

An XNNPACK FP16 candidate kept input normalization and public I/O in FP32 and
fused each Conv/BatchNorm pair before converting the network to FP16. Eager
agreement looked promising, but the actual exported runtime shifted the bicycle
mask to 0.9549 IoU (probability MAE 0.01111) against the shipped FP32 model.
It is not published either. This result is specific to the XNNPACK FP16
candidate; the published CoreML FP16 artifact was validated separately.

## Delegation

A local event-traced ExecuTorch 1.4.1 build and `executorch-profiler` measured
five FP32 executions: 99.68% of elapsed time was inside the XNNPACK delegate and
0.31% was the portable `amax` input reduction. Convolution dominated execution.
These are timing shares, not a count of delegated graph operators. Removing
the reduction would not address the main cost and would change the upstream
image-dependent input normalization.
