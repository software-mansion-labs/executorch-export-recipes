---
license: mit
---

# Introduction

This repository hosts the [EasyOCR](https://github.com/JaidedAI/EasyOCR) models — the
[CRAFT detector](https://github.com/clovaai/CRAFT-pytorch) and the [CRNN recognizer](https://www.jaided.ai/easyocr/modelhub/) — for the
[React Native ExecuTorch](https://www.npmjs.com/package/react-native-executorch) library,
exported to `.pte` for the **ExecuTorch** runtime (XNNPACK, CoreML and Vulkan backends).

If you'd like to run these models in your own ExecuTorch runtime, refer to the
[official documentation](https://pytorch.org/executorch/stable/index.html) for setup instructions.

Each language ships as **one fused `.pte`** (CRAFT *detect* + CRNN *recognize* in a single
file) per backend, with a single **dynamic** `detect` method and one fixed-width `recognize`
method (no per-size method buckets). The `.pte` is a pure tensor→tensor function; all
pre/post-processing (resize, normalize, box extraction, crop, CTC decode) is the client's job
and is driven by `config.json`. EasyOCR is the *fallback* pipeline —
[PP-OCRv6](https://huggingface.co/software-mansion/react-native-executorch-pp-ocrv6) is primary.

## Repository layout

```
<language>/<backend>/config.json                 # per-backend spec
<language>/<backend>/easy_ocr_<language>_<backend>_<precision>.pte
<language>/charset.json                           # charset[i] -> logit i+1, blank = 0
```

## Languages

| code | charset size | code | charset size |
|---|---|---|---|
| english | 96 | korean | 1008 |
| latin | 351 | telugu | 165 |
| japanese | 2214 | kannada | 167 |
| zh_sim | 6718 | cyrillic | 207 |

All languages share the same CRAFT detector and CRNN architecture — they differ **only** in
the recognizer charset. The detector half of each fused PTE is identical across languages.
Charset index `i` maps to logit `i + 1` (logit `0` is the CTC blank).

## Methods & I/O contract

| method | input | output |
|---|---|---|
| `detect` (CRAFT) | `[1,3,H,W]` f32 RGB, **ImageNet-normalized by the client**: `(x/255 − mean)/std`, `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` | score `[1,H/2,W/2,2]` (region + affinity, NHWC) |
| `recognize` (CRNN) | `[1,3,64,512]` f32 RGB, client-normalized `(x/255 − 0.5)/0.5` (RGB→gray conv is baked) | `[1,127,V]` probs (softmax baked) |

**Nothing is baked for input normalization** — the client normalizes before calling, with
*different* norms per method (ImageNet for detect, `0.5/0.5` for recognize). `detect` exports
the detection heatmap only; CRAFT's RefineNet feature map is dropped, as nothing on-device
consumes it.

## Shape discovery (`get_model_schema`)

Every `.pte` exports one no-arg constant method, **`get_model_schema`**, returning a JSON
`ModelSpec` string: per method, the input and output parameter specs (dtype plus a domain per
dimension — `constant`, `range` with `{min, max, step}`, or `enum` with explicit `choices`) and
the runtime constraints the method declares over its dimensions. The older
`get_dynamic_dims_<m>` / `get_enum_shapes_<m>` companion methods are **gone** — everything they
carried now lives in this one document.

| backend | `detect` H | `detect` W |
|---|---|---|
| `xnnpack` | `range` `[320, 1280]` step 32 | `range` `[320, 1280]` step 32 |
| `vulkan` | `range` `[320, 1280]` step 32 | `range` `[320, 1280]` step 32 |
| `coreml` | `enum` `320, 800, 1280` | `enum` `320, 800, 1280` |

`recognize` is fixed at `[1,3,64,512]` on every backend and declares a **linear runtime
constraint** tying its input width to its CTC timestep count: `width = 4 × timesteps + 4`. The
CRNN crops a trailing timestep, so `512 → 127` is *not* a plain width/timestep ratio — read the
constraint rather than dividing. `detect` runs once per image; `recognize` runs once per text
line, with every crop snapped to width 512 (the BiLSTM only delegates at a fixed time dimension).

## Backends

| backend | target | detect | recognize | warm latency (detect @800² / recognize) |
|---|---|---|---|---|
| `xnnpack` | CPU | int8, dynamic (see note) | int8 @512 | ~810 ms / ~24 ms (Galaxy S24) |
| `coreml` | Apple ANE | weight-only int8, enumerated | weight-only int8 @512 | ~83 ms / ~27 ms (Apple M-series ANE) |
| `vulkan` | Android GPU | fp16, dynamic (resize) | int8 @512 on **XNNPACK** (mixed-delegate) | ~750 ms / ~24 ms (Galaxy S24, Xclipse 940) |

> **XNNPACK detect accuracy note:** the int8 detector is calibrated for sizes **≤ 800 px**
> (its accurate operating band). Larger inputs up to 1280 are accepted but **best-effort** —
> static-activation int8 is not stable at ≥ 960 px (this was equally true, though unmeasured,
> of the previous per-bucket builds). Prefer resizing pages to ≤ 800 on CPU; the Vulkan and
> CoreML detectors are accurate over their full advertised ranges. The Vulkan detector's lower
> bound was 800 px until 2026-08 and is now 320, matching its width bound; that lower stretch is
> fp16 like the rest of the range but has not been parity-checked.

## CoreML notes (iOS)

- The CoreML `.pte` is a **multifunction** Core ML model (`detect` + `recognize` share one
  precompiled `.mlmodelc`). Requires **iOS 18+** and an ExecuTorch runtime ≥ 1.3.
- First-ever load on a device triggers a one-time per-shape ANE specialization (OS-cached
  afterwards) — warm each model once after install.

## Compatibility

If you intend to use these models outside of React Native ExecuTorch, make sure your runtime is
compatible with the **ExecuTorch** version used to export the `.pte` files. For more details, see
the compatibility note in the
[ExecuTorch GitHub repository](https://github.com/pytorch/executorch/blob/main/runtime/COMPATIBILITY.md).
If you work with React Native ExecuTorch, the library constants guarantee compatibility with the
runtime used behind the scenes.

These models were exported with ExecuTorch 1.3.1 and **no forward compatibility** is
guaranteed; older runtimes may not load them.
