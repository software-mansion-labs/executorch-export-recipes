# XNNPACK precision experiments

FP32 and selective INT8 are published. Full-network static INT8, an earlier
mixed INT8/FP32 recipe and FP16 were rejected because they changed the masks.

## Selective INT8 (published)

The final recipe quantizes only `stage1d.rebnconvin`, `stage1d.rebnconv1d`,
`stage2d.rebnconvin` and `stage2.rebnconvin`. These convolutions dominate host
latency. Remaining convolutions, concatenations and residual additions stay
FP32. Public I/O and normalization are unchanged.

PT2E uses per-channel symmetric INT8 weights and per-tensor affine INT8
activations with HistogramObserver. The weight observer minimum scale is
float32 epsilon. Calibration uses the same 13 pinned upstream images listed
below. Export scripts download these images automatically.

ExecuTorch 1.4.1's concatenation annotator ignores its filter argument, so
module exclusions alone do not keep skip connections in FP32. The export
selects only convolution operators, with an explicit module filter. Weight
attributes are allowed through this filter because they lack module-stack
metadata. This avoids quantizing unrelated concatenations.

Actual `.pte` execution was checked against the published FP32 `.pte` on the
three upstream images below plus ten COCO val2017 images, IDs 139, 285, 632,
724, 776, 785, 802, 872, 885 and 1000. None was used for calibration. Average
mask IoU was 0.9953, minimum 0.9868; probability MAE ranged from 0.00015 to
0.00543. All-black and all-white inputs also produced finite outputs.

| Image       | Selective INT8 mask IoU | Probability MAE |
| ----------- | ---------------------- | --------------- |
| `bike.jpg`  | 0.9947                 | 0.00158         |
| `boat.jpg`  | 0.9956                 | 0.00152         |
| `whisk.png` | 0.9992                 | 0.00055         |

On an Apple M1 Pro, 30 interleaved ExecuTorch runs after five warm-ups measured
156.0 ms median for FP32 and 87.4 ms for selective INT8 (1.78× faster). File
sizes were 4,610,948 and 4,007,684 bytes. These are host measurements, exclude
RNE preprocessing/rendering, and need on-device confirmation. This small
sample measures conversion agreement, not ground-truth segmentation accuracy.

The event-traced ExecuTorch 1.4.1 runtime and cloned profiler confirmed all
four selected convolutions execute as quantized XNNPACK kernels. Over five
real-input executions, the delegate accounted for 99.36% of method time and
the portable input `amax` for 0.63%.

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

The earlier candidate excluded convolutions in `stage1`, `stage2`, `stage1d`,
`stage2d`, `side1`, `side2`, and `outconv`, using the same calibration data.
The concatenation-filter issue described above still quantized skip connections,
including those inside excluded modules. Agreement improved, but the boat mask
still fell to 0.9410 IoU and the latency gain was small. This candidate is not
published. The selective recipe above replaces this approach.

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
