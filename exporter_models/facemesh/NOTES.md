# facemesh

MediaPipe's face-landmark model ("face mesh"): 468 3-D landmarks plus a
face-presence score, from a 192x192 crop that already contains one face. It is
the second stage of the MediaPipe face pipeline, with `blazeface` as the first.

## Source

Weights are a TFLite -> PyTorch convert of Google's `face_landmark.tflite`, done
by [tiqq111/mediapipe_pytorch](https://github.com/tiqq111/mediapipe_pytorch).
There is no release artifact, so `_facemesh/weights.py` pulls the checkpoint
from that repo at a pinned commit and verifies its sha256. The model code in
`_facemesh/facial_lm_model.py` and `_facemesh/utils.py` is vendored from the
same commit; upstream ships no LICENSE file, and the weights it converts are
Apache-2.0 (MediaPipe), which is what the config declares.

Two upstream quirks are preserved so the published state dict loads unchanged:

- the two heads are named the wrong way round (`self.confidence` regresses the
  landmarks, `self.facial_landmarks` scores them);
- landmark coordinates come out in input-pixel units (0..192 for x and y, the
  same scale for z), not normalized.

## Contract

```
forward: [1, 3, 192, 192] f32
      -> (boxes [1, 4] f32, scores [1] f32, keypoints [1, 468, 4] f32)
```

Input is RGB, CHW, normalized `v / 127.5 - 1`. This is the same
boxes/scores/keypoints shape every other keypoint model publishes, so the
runtime reuses `createKeypointDetector` rather than growing a second face
pipeline. `_facemesh/wrapper.py` bakes the decode in: the box is the xyxy hull
of the mesh, the score is the sigmoid of the presence logit, and each landmark
is `x, y, confidence, z`. There is no per-landmark confidence, so slot 2 is a
constant 1 — the same thing blazeface's export does with its six landmarks.
Slot 3 is upstream's relative depth, on the same scale as x.

## ReflectionPad2d

The 1-pixel top-left reflection pad on the input is written as two slice-concats
(`utils.reflect_pad_top_left`). It is bit-identical, but `ReflectionPad2d`
decomposes to an arange/abs/sub/index gather that the XNNPACK partitioner leaves
on the portable kernels; the concat form lowers into the delegate. With it, both
backends are a single delegate blob with nothing outside but the output tuple
unpack.

## Accuracy (ExecuTorch 1.4.1, torch 2.14.0, coremltools 9.0)

Landmark displacement against eager over 100 held-out CelebA faces — held out
from the int8 calibration set, which is the only honest way to score it
(`exports/facemesh_accuracy.py`). Pixels are on the model's own 192x192 input.

| export | size | mean | p95 | max | worst score error |
| --- | --- | --- | --- | --- | --- |
| xnnpack fp32 | 2.46 MB | 0.000 | 0.000 | 0.000 | 0.000 |
| coreml fp16  | 1.56 MB | 0.038 | 0.077 | 0.122 | 0.002 |
| xnnpack int8 | 0.74 MB | 0.621 | 1.319 | 6.308 | 0.494 |

Core ML numbers are from the host runtime; `ALL` resolves differently on device,
so treat them as a lowering check rather than a device result. Vulkan cannot be
scored on the host at all — the pip wheel has no Vulkan backend and returns
0x20 — so it is scored on device below.

## int8 is measured and rejected

int8 is a third of the size but **slower than fp32 and materially less
accurate**, so it is not published. The graph is 0.75 ms of depthwise
convolutions at 192x192: the arithmetic is not the bottleneck, so narrowing it
buys nothing while the quantize/dequantize boundaries cost real time. The worst
score error of 0.494 is the bigger problem — a face scoring 0.99 can read 0.50,
which sits exactly on the default presence threshold.

Two export traps if it is ever revisited:

- Quantizing the decode tail as well as the trunk makes the concat of the
  landmark slices with the constant confidence column pull mismatched qparams
  into one `cat`, and the XNNPACK runtime rejects the subgraph with `0x1`.
  `xnnpack/export.py` therefore quantizes `model.model` only.
- `convert_pt2e`'s GraphModule rejects `.eval()`, so the wrapper around it must
  not call it.

## Device benchmarks

Galaxy S26 Ultra (SM-S948B, Adreno 840), ET 1.4.1 `executor_runner` built from
`~/et-1.4.1/executorch` (md5 `d15c0b4b...`), real face input, 5 interleaved
rounds of 60 executions, first 10 of each round dropped, median of the rest:

| export | median | p10 | p90 | vs fp32 |
| --- | --- | --- | --- | --- |
| xnnpack fp32 | 0.754 ms | 0.650 | 0.996 | 1.00x |
| xnnpack int8 | 0.972 ms | 0.827 | 1.223 | 0.78x |
| vulkan fp16  | 1.631 ms | 1.515 | 1.959 | 0.46x |

Every round agreed on the order. Outputs were pulled back and checked against
eager on the same input rather than assumed: vulkan fp16 is correct to 0.21 px
max, so it loses on dispatch overhead and not on a defect — a sub-millisecond
graph never repays a GPU handoff. **Android ships xnnpack fp32 only.**

Core ML compute units on the host (`exports/facemesh_coreml_sweep.py`, 25
held-out faces, median of 50 warm runs on an M-series Mac):

| compute unit | xy mean | xy max | ms |
| --- | --- | --- | --- |
| `ALL` | 0.038 | 0.122 | 0.344 |
| `CPU_ONLY` | 0.256 | 1.132 | 0.787 |
| `CPU_AND_GPU` | 0.033 | 0.100 | 0.834 |
| `CPU_AND_NE` | 0.038 | 0.122 | 0.350 |

`ALL` and `CPU_AND_NE` agree to the last digit, so `ALL` resolves to the Neural
Engine here. `CPU_ONLY` is both the slowest and the least accurate, which is the
known macOS fp16 CPU path. This is a shortlist, not the answer: the compute unit
has to be confirmed on an iPhone.

No MLX export: the graph is 2.4 MB of convolutions at 192x192, and the Android
result above is the same lesson — dispatch overhead dominates at this size.

## Blocked until ExecuTorch 1.4.1

The model is PReLU-heavy, and PReLU tripped an XNNPACK use-after-free
([pytorch/executorch#17559](https://github.com/pytorch/executorch/issues/17559),
fixed by [#21480](https://github.com/pytorch/executorch/pull/21480)). The fix is
in 1.4.1 — do not re-pin this model below it.
