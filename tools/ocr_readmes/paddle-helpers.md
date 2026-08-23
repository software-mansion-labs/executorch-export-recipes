---
license: apache-2.0
---

# Introduction

This repository hosts the **PaddleOCR document helper models** — page-orientation
classification, dewarping, and table-structure recognition — for the
[React Native ExecuTorch](https://www.npmjs.com/package/react-native-executorch) library,
fused into **one multi-method `.pte`** per backend for the **ExecuTorch** runtime
(XNNPACK, CoreML, Vulkan). These are document pre/post-processing companions to
[`react-native-executorch-pp-ocrv6`](https://huggingface.co/software-mansion/react-native-executorch-pp-ocrv6) —
not an OCR model on their own.

If you'd like to run these models in your own ExecuTorch runtime, refer to the
[official documentation](https://pytorch.org/executorch/stable/index.html) for setup instructions.

## Repository layout

```
<backend>/config.json                    # per-backend spec
<backend>/paddle_helpers_<backend>_<precision>.pte
```

## Methods

The fused `.pte` exposes **four methods** (the `.pte` is pure tensor→tensor; the client
does normalization, `argmax`/`softmax`, grid-sampling and the decode loop). All methods are
**fixed-shape** — the exact input shapes below are also declared in `config.json`, and since
nothing here is dynamic the model carries no `get_model_schema` overrides; a client reads the
signatures straight from ExecuTorch's `MethodMeta`.

| method | source model | input | output | purpose |
|---|---|---|---|---|
| `orientation` | [PP-LCNet doc_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors) | `[1,3,224,224]` | `logits[1,4]` | page rotation **0 / 90 / 180 / 270°** (argmax) |
| `dewarp` | [UVDoc](https://huggingface.co/PaddlePaddle/UVDoc_safetensors) | `[1,3,712,488]` | `grid[1,2,45,31]` | sampling grid → `grid_sample` to unwarp a curved/folded page |
| `table_encode` | [SLANet_plus](https://huggingface.co/PaddlePaddle/SLANet_plus_safetensors) | `[1,3,488,488]` | `feat[1,256,96]` | encode a cropped table image (run **once**) |
| `table_decode_step` | SLANet_plus decoder | `(feat[1,256,96], hidden[1,256], onehot[1,50])` | `(probs[1,50], hidden[1,256])` | one **autoregressive** structure-token step |

The table decode loop is data-dependent, so it stays on the client: run `table_encode` once,
then call `table_decode_step` per token, feeding the previous step's hidden state back in and
the chosen token back as a one-hot. The first step takes an **all-zero** one-hot — the exported
decoder embeds its own start token.

## Backends, precision & latency (warm)

| backend | target | precision | size | orientation / table_encode / dewarp / decode_step |
|---|---|---|---|---|
| `xnnpack` | CPU | **dynamic int8** (qd8 — lossless; static int8 is lossy on the regressors) | ~28 MB | 2.0 / 30 / 209 / 0.21 ms (Galaxy S24) |
| `coreml` | Apple ANE | weight-only int8 | **11.9 MB** | 0.3 / 1.4 / 4.7 / 0.13 ms (Apple M-series ANE) |
| `vulkan` | Android GPU | fp16, except the table path → XNNPACK (mixed-delegate) | 23 MB | 6.0 / 17 / 66 / 0.21 ms (Galaxy S24) |

- The GPU wins on the heavy CNNs (dewarp 3×); the tiny `orientation` and the dispatch-bound
  autoregressive `table_decode_step` are faster on CPU — hence the Vulkan build routes the
  table path to XNNPACK.
- `table_decode_step` is always computed at full precision on CPU (autoregressive stability).

## Compatibility

If you intend to use these models outside of React Native ExecuTorch, make sure your runtime is
compatible with the **ExecuTorch** version used to export the `.pte` files. For more details, see
the compatibility note in the
[ExecuTorch GitHub repository](https://github.com/pytorch/executorch/blob/main/runtime/COMPATIBILITY.md).
If you work with React Native ExecuTorch, the library constants guarantee compatibility with the
runtime used behind the scenes.

These models were exported with ExecuTorch 1.3.1 and **no forward compatibility** is
guaranteed; older runtimes may not load them.
