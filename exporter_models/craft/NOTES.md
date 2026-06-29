# craft notes

CRAFT OCR text-detection, 2 variants (fp32 + int8) via the multi-pte path.
`_craft.py` is the recipe's model code (`exports/ocr/export/craft.py`), cv2/executorch
top-imports stripped (unused on this path); `export.py` transcribes the build + quant.

## Two intentional adaptations from the recipe

1. **Methods: `forward_1280` + `forward_320` only (NOT `forward_800`).** The recipe
   builds 3 methods, but the PUBLISHED HF config (`xnnpack/config.json`) declares only
   1280 and 320. The harness verifies against the published contract, so producing
   exactly those two is correct; `forward_800` is not in the published artifact. Both
   variants verified `io_ok`.

2. **int8 calibration: synthetic data.** The recipe calibrates int8 from
   `calibration_images/*.png` via cv2, falling back to synthetic (`randn*0.5`) when the
   images are absent. The harness uses the synthetic fallback unconditionally — no cv2 /
   no vendored images. Calibration data does not affect the `.pte` I/O contract (only
   weight ranges), so this is behavior-equivalent for the harness's verification; it may
   differ slightly in int8 accuracy from the image-calibrated published .pte.
