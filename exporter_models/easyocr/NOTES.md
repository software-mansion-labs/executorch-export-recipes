# EasyOCR (fused, sized-by-language) — export notes

Fused **CRAFT detector + CRNN recognizer** OCR. ONE PTE per (language, backend):

- `detect [1,3,H,W]` — **true-dynamic**: H in `[800, 1280]`, W in `[320, 1280]`,
  step 32 (CRAFT /32), with a `get_dynamic_dims_detect` companion (no-arg method
  returning `int32 [4, 3]` rows `[min, max, step]`; static dims `[n, n, 1]`).
  CoreML has no RangeDim → enumerated (800²/1280²/1280×320) + a
  `detect_enum_shapes` constant (`int32 [3, 4]` NCHW) instead of the companion.
- `recognize [1,3,64,512]` — ONE **fixed width** (512 on all backends), declared
  via a `recognize_enum_shapes` constant (`int32 [1, 4]`); the C++ snaps every
  crop to it. The CRNN's BiLSTM cannot be dynamic on CPU/ANE (see below).

This **supersedes** the old split `easyocr/craft` + `easyocr/crnn` stubs — the
fused, per-language model is canonical. EasyOCR is the *fast fallback*
(PP-OCR is primary); the fixed-width recognizer is that rigid profile.

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
`recognize` output shape is `[1, W/4-1, V]`. V is fixed by the gen2 checkpoint:

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

The CRNN is exported **unrolled** at the fixed width (`nn.LSTM` specializes the
time dim → fully delegates at concrete `T = W/4-1 = 127`). The dynamic-time-dim
`scan`-based BiLSTM (`arch.ScanBiLSTM` / `ScanBiL`) is kept in the vendored arch
for a future *true-dynamic* recognizer (runs portable CPU — `scan` is not a
delegate op), but the shipped fixed-width recognizer does not use it.

## Per-backend precision + delegate

| backend | CRAFT detect | CRNN recognize @512 | notes |
|---|---|---|---|
| xnnpack | **static-act int8** dynamic, H,W ∈ [320, 1280], calibrated at ≤800 only (norm folded into conv1) | int8 static (per-channel, real-img calib) | all XNNPACK CPU. Static int8 scales hold in the ≤800 band (corr 0.970–0.9997, 4 imgs × 5 shapes) — **above ~800 is BEST-EFFORT by deliberate decision** (~0.55–0.97 content-dependent at 1280²; no calibration fixes it, and both the OG dynamic AND the old bucketed detect_1280 were equally broken there, measured on the shipped artifacts). qs8 chains = bucketed-class speed: S24 detect warm **811ms@800² / 513@640² / 336@320×800 / ~2100@1280²** (bucketed 731ms; qd8 alt — correct@1280 — was 1296ms; fp16/fp32 are 2106/4214ms, fp16 is fake on XNNPACK). Norm fold dodges an ET delegate bug (quantize convert fed by in-blob sub/div fails xnn_reshape_runtime). The qd8 machinery (is_relu_node annotate patch, exclude_modules) stays in backends/xnnpack.py — flip to qd8 if correct 1280 CPU detect ever matters more than 485ms. |
| coreml  | weight-only int8, enumerated | weight-only int8, static | fp16 compute (ANE); detect full-graph (enumerated requires it). |
| vulkan  | fp16 dynamic via resize (**Vulkan** GPU) | int8 (**XNNPACK** CPU) | MIXED-delegate. int8 q8ta cannot resize → the dynamic contract forces fp16 (fp32 graph + `force_fp16`; the old static q8ta int8 was smaller+faster but rigid). CRNN BiLSTM won't delegate to Vulkan → XNNPACK. `texture_limits=(2048,2048,2048)` (Xclipse 3D-texture cap). |

## Calibration

int8 MUST be calibrated on real activations — random noise passes random-parity
but breaks on real text (see int8-calibration-real-data memory). Real crops are
vendored under `_easyocr/calibration_images/` (CRNN slices each into 4 text-band
regions; CRAFT resizes whole pages).
