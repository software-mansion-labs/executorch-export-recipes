# OCR dynamic-contract vs bucketed — full bench (2026-07-13)

Device: Samsung S24 (SM-S921B, Exynos 2400 / Xclipse 940), `runner_arm64`, warm = min of 8.
Accuracy: prob-map corr (+ bin-IoU@0.3 where noted) vs eager fp32 on real test images
(PP: `_pp/calib_data` page; EasyOCR: `_easyocr/calibration_images` 2/4). Vulkan accuracy
measured **on-device** via `runner_dump_arm64`; XNNPACK measured locally (same PTEs).
"ships" = the current converge-branch dynamic-contract artifacts.

## PP-OCRv6

| variant | quant | MB | delegation | detect warm | detect accuracy | recognize |
|---|---|---|---|---|---|---|
| **XNNPACK dynamic (ships)** | det+rec fp32 | 31.1 | det 90.9%/5b, rec 71.6%/24b | 250ms@640, 574@960 | **corr 1.0000 / IoU 1.000 at 640, 960, 1280, 1280×640** | 28ms@W320, agree 1.000 |
| XNNPACK bucketed (old) | det int8-static (per-bucket calib) + rec fp32 | 24.8 | – | 128@640, 283@960, 498@1280 | 0.985/0.93@640, **0.908/0.51@960, 0.706/0.003@1280 — BROKEN** | 22ms, agree 1.000 |
| **Vulkan dynamic (ships)** | det fp16 (resize) + rec fp32 (XNNPACK, mixed) | 26.2 | det 99.7%/**1 blob** | **34@640, 73@960** | **0.9999 / 0.989 on-device @1280** | 27ms |
| Vulkan bucketed (old) | det fp16 static + rec fp32 (XNNPACK) | 26.7 | – | 33@640, 72@960 | 0.9988 / 0.875 on-device @960 | 23ms |

## EasyOCR (english)

| variant | quant | MB | delegation | detect warm | detect accuracy (corr) | recognize |
|---|---|---|---|---|---|---|
| **XNNPACK dynamic (ships)** | det static int8, H,W∈[320,1280], calib ≤800 (norm folded) + rec int8@512 | ~33 | det ~95%/4b | **811@800**, 513@640, 336@320×800, ~2100@1280 | **0.970–0.9997 at ≤800 (4 imgs × 5 shapes)**; >800 BEST-EFFORT by decision (~0.55–0.97 @1280², same as OG dynamic + old bucketed_1280) | 24ms, agree 0.992 |
| OG dynamic (Jun 26 artifact) | det static int8 (1280-only calib, no baked norm) + rec int8@256 | 31.9 | – | 807@800, 514@1280×320, 2102@1280 | 0.988@800, **0.554@1280², 0.78–0.93@portrait** | 12ms@256 |
| XNNPACK qd8 alt (in-tree, off) | det qd8 (conv1_1/1_2+cls fp32), range [320,1280] | 32.9 | det 94.2%/4b | 1296@800, 3392@1280 | 0.9995@800, **0.982@1280 — the only correct-at-1280 int8** | 24ms |
| XNNPACK bucketed (old) | det int8-static ×3 buckets + rec int8 ×4 widths | 34.3 | – | **731@800**, 459@1280×320, 2132@1280 | 0.993@800, **0.569@1280 — BROKEN**, 0.958@1280×320 | 26ms, agree 0.992 |
| **Vulkan dynamic (ships)** | det fp16 (resize) + rec int8@512 (XNNPACK, mixed) | 53.4 | det 98.2%/1b | 752@800, ~2000@1280 | **0.9991 on-device @1280** | 24ms |
| Vulkan bucketed (old) | det q8ta int8 static + rec int8 (XNNPACK) | 34.4 | – | **278@800**, 684@1280 | 0.982 on-device @800 (1280 untested; same int8-static class as above) | 27ms |

## EasyOCR XNNPACK detect@800² precision ladder (S24, the "sacrifice quant?" question)

| precision | warm | size | accuracy |
|---|---|---|---|
| bucketed static int8 (rigid) | 731ms | ~21MB | broken @1280 bucket |
| full-qd8 dynamic | 943ms | 20.8MB | broken @1280² (0.70) |
| **qd8, conv1_1/1_2+cls fp32 (ships)** | **1296ms** | 21MB | ≥0.982 everywhere |
| fp16 dynamic | 2106ms | 83MB | ~1.0 |
| fp32 dynamic | 4214ms | 83MB | 1.0 |

On XNNPACK, **quant IS the speed** (memory-bound convs): dropping quantization is 1.6–3.3×
slower. fp16 is fake on XNNPACK (fp32-sized blob, fp32 speed).

## Conclusions

1. **PP: dynamic wins outright.** Vulkan dynamic-resize is free (34/73ms = bucketed speed,
   1 blob); XNNPACK fp32 is 2× slower than bucketed int8 but bit-exact — and the bucketed
   int8 detect was computing garbage at 960/1280 anyway.
2. **Static-activation int8 conv detect is broken at ≥960px regardless of contract** —
   per-bucket calibration never fixed it; nobody had parity-checked the big buckets before.
3. **EasyOCR XNNPACK: SOLVED via narrow-band static int8** — one scale set is stable
   within [320, 800] (corr 0.970–0.9997) at qs8-chain speed: **811ms@800² ≈ bucketed's
   731ms**, truly dynamic in-band. The wide-range alternatives: qd8 1296ms, fp16 2106ms,
   fp32 4214ms. Band ceiling is real: [320,960] wobbles at 960² (0.918).
4. **EasyOCR Vulkan is the one remaining regression: 752 vs 278ms** — the old static q8ta
   int8 CRAFT can't resize; flipping `supports_resize` on the q8ta ops exports and runs but
   SILENTLY computes at the export shape (verified by shape-independent timing + dump) —
   real int8 dynamic needs upstream Vulkan shader reshape work. Option: add a static q8ta
   `detect_800` fast-path method alongside the dynamic fp16 `detect` in one PTE.
5. Recognize is a wash everywhere (22–28ms, agree ≥0.992).

Caveat: accuracy = prob-map parity on 2 images; the definitive word would be a box-level
end-to-end eval, but 0.57–0.71-corr maps do not produce usable boxes under any postproc.
