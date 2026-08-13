---
license: apache-2.0
---

# Introduction

This repository hosts [PaddleOCR PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) (the
[detector](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_safetensors) +
[recognizer](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_safetensors)) for the
[React Native ExecuTorch](https://www.npmjs.com/package/react-native-executorch) library,
exported to `.pte` for the **ExecuTorch** runtime (XNNPACK, CoreML and Vulkan backends).

If you'd like to run these models in your own ExecuTorch runtime, refer to the
[official documentation](https://pytorch.org/executorch/stable/index.html) for setup instructions.

PP-OCRv6 is the **primary** OCR pipeline — smallest and fastest. It ships as **one fused
`.pte`** per backend with a single **dynamic** `detect` and `recognize` method each (no
per-size method buckets). The `.pte` is a pure tensor→tensor function; all pre/post-processing
(resize, normalize, DBNet box decode, perspective crop, CTC decode) is the client's job and is
driven by `config.json`. One model covers **all languages** (18 709-entry multilingual charset).

## Repository layout

```
<backend>/config.json                    # per-backend spec (see the schema link inside)
<backend>/pp_ocrv6_<backend>_<precision>.pte
charset.txt                              # 18 709 entries; charset[i] -> logit i+1, blank = 0
```

## Methods & I/O contract

| method | input | output |
|---|---|---|
| `detect` (DBNet) | `[1,3,H,W]` f32 RGB, **ImageNet-normalized by the client**: `(x/255 − mean)/std`, `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` | `[1,1,H,W]` probability map (sigmoid baked) |
| `recognize` (SVTR) | `[1,3,48,W]` f32 RGB, client-normalized `(x/255 − 0.5)/0.5` | `[1,W/8,18710]` probs (softmax baked); `charset[i]` → logit `i+1`, blank = 0 |

**Nothing is baked for input normalization** — the client normalizes before calling.
Note the two methods use *different* norms (ImageNet for detect, `0.5/0.5` for recognize).

## Shape discovery (`get_model_schema`)

Every `.pte` exports one no-arg constant method, **`get_model_schema`**, returning a JSON
`ModelSpec` string: per method, the input and output parameter specs (dtype plus a domain per
dimension — `constant`, `range` with `{min, max, step}`, or `enum` with explicit `choices`) and
the runtime constraints the method declares over its dimensions. Methods absent from the JSON
are fully static and described by ExecuTorch's own `MethodMeta`. The schema is validated
against
[`config.schema.json`'s `ModelSpec`](https://huggingface.co/software-mansion/react-native-executorch-spec)
counterpart in the library at load time. The older `get_dynamic_dims_<m>` / `get_enum_shapes_<m>`
companion methods are **gone** — everything they carried now lives in this one document.

| backend | `detect` H, W | `recognize` W (H fixed 48) |
|---|---|---|
| `xnnpack`, `vulkan` | `range` `[640, 1280]` step 32 | `range` `[160, 1280]` step 8 |
| `coreml` | `enum` `640, 960, 1280` per dimension | `enum` `160, 320, 480, 640, 1280` |

`recognize` additionally declares a **linear runtime constraint** tying its input width to its
CTC timestep count — `width = 8 × timesteps + 0` — so a client can size the probs output for
whatever width it picks instead of inferring a ratio.

## Backends

| backend | target | detect | recognize | warm latency (detect @960² / recognize) |
|---|---|---|---|---|
| `xnnpack` | CPU | static int8, true-dynamic | fp32, true-dynamic | ~28 ms recognize (Galaxy S24) |
| `coreml` | Apple ANE | weight-only int8, enumerated | weight-only int8, enumerated | ~12–15 ms / ~2 ms (Apple M-series ANE) |
| `vulkan` | Android GPU | fp16, true-dynamic (resize) | fp32 on **XNNPACK** (mixed-delegate) | ~73 ms / ~27 ms (Galaxy S24, Xclipse 940) |

> **Vulkan is mixed-delegate**: DBNet detects on the GPU, the SVTR recognizer runs on the CPU
> (XNNPACK) — the 18 709-token vocab head is not Vulkan-safe, and int8 SVTR is lossy, so the
> recognizer stays fp32 on CPU for correctness.

> **Why int8 detect but fp32 recognize?** The detector is static-activation int8, calibrated on
> real pages at 1280. It was previously shipped as fp32 on the belief that int8 was unstable
> across dynamic input sizes; re-measured in 2026-08 that no longer holds (correlation 0.956 to
> 0.994 against eager fp32 from 640 to 1280 square, best at 1280), and on-device the int8 build
> returns the same detections about 15% faster end to end at 23.9 MB instead of 31.1 MB. The
> SVTR recognizer stays fp32: int8 is lossy on its attention stack.

## CoreML notes (iOS)

- The CoreML `.pte` is a **multifunction** Core ML model (`detect` + `recognize` share one
  precompiled `.mlmodelc`). Requires **iOS 18+** and an ExecuTorch runtime ≥ 1.3 (multifunction
  loading via `functionName`).
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
