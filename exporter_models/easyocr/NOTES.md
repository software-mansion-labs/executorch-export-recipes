# EasyOCR (fused, sized-by-language) — export notes

Fused **CRAFT detector + CRNN recognizer** OCR. ONE PTE per (language, backend),
**static bucketed** methods (`detect_<S>` / `detect_<W>x<H>` / `recognize_<W>`)
plus an `is_bucketed()` constant returning `int32[2,N]` (row 0 = CRAFT square
detector sides, row 1 = CRNN recognizer widths, `-1` padded). The C++ runtime
snaps each input to the nearest bucket.

This **supersedes** the old split `easyocr/craft` + `easyocr/crnn` stubs — the
fused, per-language model is canonical. EasyOCR is the *rigid fast fallback*
(PP-OCR is primary); the bucketed deliverable is that rigid profile.

## Sizes = languages

`sizes: [english, latin, japanese, zh_sim, korean, telugu, kannada, cyrillic]`.
Config lives at `<lang>/<backend>/config.json`; there is one `export.py` per
backend at the model root (`<backend>/export.py`), shared across all languages —
it reads `lang = config["size"]` and calls `_easyocr.builder.build_and_lower(backend, lang, out)`.

## Weights (never committed)

Fetched at export time from the JaidedAI EasyOCR GitHub releases (same as
`download_weights.sh`), cached under `~/.cache/exporter/easyocr/`. See
`_easyocr/weights.py`:

- CRAFT detector: `pre-v1.1.6/craft_mlt_25k.zip` (language-independent, VGG16-bn backbone).
- per-lang CRNN gen2: `english_g2` / `latin_g2` / `japanese_g2` / `zh_sim_g2` /
  `korean_g2` / `cyrillic_g2` (v1.3 / v1.6.1), `telugu` / `kannada` (v1.2).

## Per-language vocab V (recognizer output last dim)

`V = len(charset) + 1` (CTC blank at logit 0; `charset[i]` → logit `i+1`). The
`recognize_<W>` output shape is `[1, W/4-1, V]`. V is fixed by the gen2 checkpoint:

| lang | charset | V |
|---|---|---|
| english | 96 | 97 |
| latin | 351 | 352 |
| japanese | 2214 | 2215 |
| zh_sim | 6718 | 6719 |
| korean | 1008 | 1009 |
| telugu | 165 | 166 |
| kannada | 167 | 168 |
| cyrillic | 207 | 208 |

The charset itself is emitted at export time as `EasyOCR_<lang>.charset.txt` (JSON
list) beside the PTE, sourced from the installed `easyocr` package
(`easyocr.config.recognition_models["gen2"][<key>]["characters"]`). Emission is
best-effort — V (above) is hardcoded in `_easyocr/weights.py`, so the PTE never
depends on the `easyocr` package.

## Baked unified runtime contract

- **Detector** — CRAFT's ImageNet norm (`mean=[0.485,0.456,0.406]`,
  `std=[0.229,0.224,0.225]`) is BAKED (`DetNorm`), so the runtime feeds raw
  `[0,1]` RGB — same input contract as PaddleOCR's DBNet. Output is a tuple:
  score `[1,H/2,W/2,2]` (region+affinity) and feature `[1,32,H/2,W/2]`.
  NOTE: the original `export_easyocr.py` omitted `DetNorm` on the *vulkan*
  CRAFT branch; here it is baked on all three backends for contract consistency.
- **Recognizer** — a fixed RGB→gray 1×1 conv (cv2 weights `0.299/0.587/0.114`) +
  output softmax are BAKED (`UnifiedCRNN`), so the CRNN takes `[1,3,64,W]` RGB and
  emits `[1,T,V]` **probs** — same contract as SVTR. Lossless (gray weights sum to
  1, commute with the `(x-0.5)/0.5` norm, reproduce cv2 grayscale to ~3e-9).

## CRNN time dim / unrolling / scan

The bucketed CRNN is exported **unrolled** per width (`nn.LSTM` specializes the
time dim → fully delegates at concrete `T = W/4-1`). The dynamic-time-dim
`scan`-based BiLSTM (`arch.ScanBiLSTM` / `ScanBiL`) is kept in the vendored arch
for the alternative *true-dynamic* deliverable (runs portable CPU — `scan` is not
a delegate op), but the shipped bucketed fallback does not use it.

## Per-backend precision + delegate

| backend | CRAFT detect | CRNN recognize | notes |
|---|---|---|---|
| xnnpack | int8 (per-tensor, real-img calib) | int8 (per-channel, real-img calib) | all XNNPACK CPU. |
| coreml  | weight-only int8 | weight-only int8 | fp16 compute (ANE); static full-graph per method. |
| vulkan  | int8 per-channel (**Vulkan** GPU) | int8 (**XNNPACK** CPU) | MIXED-delegate. q8ta CRAFT needs per-channel weights + `force_fp16=False` (smaller+faster than fp16 on a memory-bound net). CRNN BiLSTM won't delegate to Vulkan → XNNPACK. `texture_limits=(2048,2048,2048)` (Xclipse 3D-texture cap). |

## Calibration

int8 MUST be calibrated on real activations — random noise passes random-parity
but breaks on real text (see int8-calibration-real-data memory). Real crops are
vendored under `_easyocr/calibration_images/` (CRNN slices each into 4 text-band
regions; CRAFT resizes whole pages).
