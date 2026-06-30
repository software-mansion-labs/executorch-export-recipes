# PP-OCRv6 (paddleocr) — export notes

Fused detect + recognize OCR. One PTE per backend, **static bucketed** methods
(`detect_<S>` / `recognize_<W>`) plus an `is_bucketed()` constant returning
`int32[2,N]` (row 0 = detector square sizes, row 1 = recognizer widths). The C++
runtime snaps each input to the nearest bucket.

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
| xnnpack | int8 (real-page calib) | **fp32** | int8 rec strands `dequantize_per_channel.out` (not in the ET runtime) → method fails to load; keep fp32. |
| coreml  | weight-only int8 | weight-only int8 | fp16 compute (ANE). |
| vulkan  | fp16 (Vulkan) | fp32 (**XNNPACK**) | MIXED-delegate: SVTR vocab head crashes Vulkan `make_ivec2`; `texture_limits=(2048,2048,2048)` caps the V=18710 projection. |

## Calibration

Real images are vendored under `_pp/calib_data/` (multilingual OCR crops/pages).
int8 calibration on random noise passes random-parity but **breaks on real input**
(see the int8-calibration memory) — keep real data.
