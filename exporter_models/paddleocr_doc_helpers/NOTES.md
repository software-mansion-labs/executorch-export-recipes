# PaddleOCR doc-helpers (paddleocr_doc_helpers) — export notes

Three PaddleOCR *supporting* models fused into ONE PTE per backend, exposing **4
methods** (NOT bucketed, no `is_bucketed()`):

| method | input(s) | output(s) |
|---|---|---|
| `orientation`       | `[1,3,224,224]` f32 | `[1,4]` f32 |
| `dewarp`            | `[1,3,712,488]` f32 | `[1,2,45,31]` f32 |
| `table_encode`      | `[1,3,488,488]` f32 | `[1,256,96]` f32 |
| `table_decode_step` | (`[1,256,96]`, `[1,256]`, `[1,50]`) f32 | (`[1,50]`, `[1,256]`) f32 |

`orientation` = PP-LCNet doc-orientation (4 classes), `dewarp` = UVDoc grid,
`table_encode` = SLANet_plus backbone feature, `table_decode_step` = one bit-exact
AR step of SLANet's structure decode (the loop itself is data-dependent → runs in
C++; one step per call).

## Prerequisites

- **PaddlePaddle transformers fork** — the helpers load via
  `transformers.PPLCNetForImageClassification`, `UVDocModel`,
  `SLANetForTableRecognition`, which only exist in the vendored fork (an
  environment prerequisite, not a pip extra). Weights are pulled from
  `PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors`, `PaddlePaddle/UVDoc_safetensors`,
  `PaddlePaddle/SLANet_plus_safetensors`.
- `cv2` (opencv) + `numpy` for the int8 calibration loader.
- `extras`: `quant` (torchao PT2E for the dynamic-int8 paths), `coreml` (coremltools).

## Per-backend precision (backend alone fixes the lossless-maximal precision)

| backend | precision | notes |
|---|---|---|
| xnnpack | **dynamic int8** (qd8) | int8 weights + runtime-quantized activations — lossless AND ~halves size vs fp32. fp16 is a no-op on XNNPACK. |
| coreml  | **weight-only int8** | fp16 compute on the ANE; immune to calibration drift. |
| vulkan  | **fp16** | CNN methods (orientation, dewarp) on the GPU; the WHOLE table path (`table_encode` + `table_decode_step`) is routed to XNNPACK via MIXED-delegate — fp16 GPU `table_encode` drifts the AR decode on hard tables, and a tiny recurrent step is dispatch-bound on a GPU. |

`table_decode_step` stays **fp32 on every backend** — quant/fp16 error compounds
across the AR loop.

## Calibration

A small flat set of real OCR document crops is vendored under
`_helpers/calib_images/`. int8 calibration on random noise passes random-parity but
**breaks on real input** (see the int8-calibration memory) — keep real data.
