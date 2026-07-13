# PP-OCRv6 (paddleocr) — export notes

Fused detect + recognize OCR. One PTE per backend with **true-dynamic** methods:

- `detect [1,3,H,W]` — H, W dynamic in `[640, 1280]` step 32 (DBNet /32).
- `recognize [1,3,48,W]` — W dynamic in `[160, 1280]` step 8 (SVTR width /8).
- `get_dynamic_dims_detect` / `get_dynamic_dims_recognize` — no-arg companions
  returning one `int32 [rank, 3]` tensor per tensor input of the target method,
  rows `[min, max, step]` (static dims `[n, n, 1]`) — the runtime validation
  contract (`exporter.dynamic_dims.build_dynamic_dims_program`).

CoreML has no RangeDim, so it ships the same two methods **enumerated** (detect:
640/960/1280 square + 1280×640 portrait; recognize: widths 160–1280) with
`detect_enum_shapes` / `recognize_enum_shapes` constants (`int32 [N, 4]` NCHW)
instead of companions — the C++ snaps to the nearest enumerated shape.

## Prerequisites

- **PaddlePaddle transformers fork** — the detector/recognizer load via
  `transformers.PPOCRV6SmallDetForObjectDetection` / `...RecForTextRecognition`,
  which only exist in the vendored fork (an environment prerequisite, not a pip
  extra). Weights are pulled from `PaddlePaddle/PP-OCRv6_small_{det,rec}_safetensors`.
- `cv2` (opencv) + `numpy` for the int8 calibration loader.
- `extras`: `quant` (torchao PT2E for the XNNPACK detector int8), `coreml`
  (coremltools).

## Baked into the graph (the unified runtime contract)

- **Detector input norm** — standard ImageNet on RGB (`mean=[0.485,0.456,0.406]`,
  `std=[0.229,0.224,0.225]`), baked. Runtime feeds raw `[0,1]` RGB. (This is the
  same norm as CRAFT, so the detector input unifies across PaddleOCR + EasyOCR.
  The old `(x-0.5)/0.5` was the *recognizer's* norm — wrong for detect.)
- **DBNet** emits probabilities (sigmoid already in the head); **SVTR** emits
  probs `[1,T,V]` (softmax in the head). `charset[i]` → logit `i+1`, blank = 0.

## Per-backend precision

| backend | detect | recognize | notes |
|---|---|---|---|
| xnnpack | **fp32** dynamic | **fp32** dynamic | STATIC-activation int8 quant params don't transfer across dynamic scales (det corr 0.59-0.88, IoU~0 — even at the calib shape; PPLCNet noise compounds through depth + sigmoid head). qd8 recovers only to ~0.95@1280 (not shippable), fp16 doesn't shrink the XNNPACK blob (stays fp32-sized). fp32 = bit-exact at every shape, ~31MB fused. int8 rec additionally wrecks the attention SVTR. |
| coreml  | weight-only int8 | weight-only int8 | fp16 compute (ANE). |
| vulkan  | fp16 (Vulkan) | fp32 (**XNNPACK**) | MIXED-delegate: SVTR vocab head crashes Vulkan `make_ivec2`; `texture_limits=(2048,2048,2048)` caps the V=18710 projection. |

## Calibration

Real images are vendored under `_pp/calib_data/` (multilingual OCR crops/pages).
int8 calibration on random noise passes random-parity but **breaks on real input**
(see the int8-calibration memory) — keep real data.
