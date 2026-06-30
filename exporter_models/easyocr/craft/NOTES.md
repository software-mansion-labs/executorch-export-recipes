# craft notes

CRAFT OCR text-detection, 2 variants (fp32 + int8) via the multi-pte path.
`_craft.py` is the recipe's model code (`exports/ocr/export/craft.py`), cv2/executorch
top-imports stripped (unused on this path); `export.py` transcribes the build + quant.

## Two intentional adaptations from the recipe

1. **Methods: `forward_1280` (1280×1280) + `forward_320` (320×320) only.** The recipe
   builds 3 methods with sizes `(1280,1280)`, `(800,800)`, `(1280,320)`, but the
   PUBLISHED HF config (`xnnpack/config.json`) declares only two: `forward_1280` =
   `[1,3,1280,1280]` and `forward_320` = `[1,3,320,320]` (SQUARE 320, not the recipe's
   landscape 1280×320). The harness verifies against the published contract, so it
   produces exactly those two square inputs — both verified `io_ok`. `forward_800` and
   the recipe's landscape 320 are not in the published artifact.

2. **int8 calibration: synthetic data.** The recipe calibrates int8 from
   `calibration_images/*.png` via cv2, falling back to synthetic (`randn*0.5`) when the
   images are absent. The harness uses the synthetic fallback unconditionally — no cv2 /
   no vendored images. Calibration data does not affect the `.pte` I/O contract (only
   weight ranges), so this is behavior-equivalent for the harness's verification; it may
   differ slightly in int8 accuracy from the image-calibrated published .pte.
