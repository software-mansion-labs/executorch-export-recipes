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

`forward: [1, 3, 192, 192] f32 -> (landmarks [468, 3] f32, score [1] f32)`

Input is RGB, CHW, normalized `v / 127.5 - 1`. `_facemesh/wrapper.py` bakes the
`1x1404x1x1 -> 468x3` reshape and the score's sigmoid into the graph, so the
`.pte` publishes the shapes the runtime consumes.

## ReflectionPad2d

The 1-pixel top-left reflection pad on the input is written as two slice-concats
(`utils.reflect_pad_top_left`). It is bit-identical, but `ReflectionPad2d`
decomposes to an arange/abs/sub/index gather that the XNNPACK partitioner leaves
on the portable kernels; the concat form lowers into the delegate. With it, both
backends are a single delegate blob with nothing outside but the output tuple
unpack.

## Verified (ExecuTorch 1.4.1, torch 2.14.0, coremltools 9.0)

Against eager on `facial_landmarks/4.jpg` from the source repo, letterboxed to
192:

| backend | file | size | score (eager 0.999991) | max xy error | max z error |
| --- | --- | --- | --- | --- | --- |
| xnnpack fp32 | `facemesh_xnnpack_fp32.pte` | 2.46 MB | 0.999991 | 0.0001 px | 0.0000 |
| coreml fp16  | `facemesh_coreml_fp16.pte`  | 1.54 MB | 1.000000 | 0.0831 px | 0.0557 |

Core ML numbers are from the host runtime; `ALL` resolves differently on device,
so treat them as a lowering check rather than a device result.

No MLX or Vulkan export: the graph is 2.4 MB of convolutions at 192x192, so the
dispatch overhead of either accelerator is not worth a sweep.

## Blocked until ExecuTorch 1.4.1

The model is PReLU-heavy, and PReLU tripped an XNNPACK use-after-free
([pytorch/executorch#17559](https://github.com/pytorch/executorch/issues/17559),
fixed by [#21480](https://github.com/pytorch/executorch/pull/21480)). The fix is
in 1.4.1 — do not re-pin this model below it.
