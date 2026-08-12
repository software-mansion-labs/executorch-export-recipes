---
license: apache-2.0
---

# Introduction

This repository hosts [PaddleOCR PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors),
an RT-DETR-based **document layout detector** (~33M params), for the
[React Native ExecuTorch](https://www.npmjs.com/package/react-native-executorch) library,
exported to `.pte` for the **ExecuTorch** runtime (XNNPACK, CoreML, Vulkan). It finds and
classifies document regions — titles, paragraphs, tables, figures, formulas, headers/footers,
etc. — and is a companion to
[`react-native-executorch-pp-ocrv6`](https://huggingface.co/software-mansion/react-native-executorch-pp-ocrv6).

If you'd like to run these models in your own ExecuTorch runtime, refer to the
[official documentation](https://pytorch.org/executorch/stable/index.html) for setup instructions.

The `.pte` is a pure tensor→tensor function; pre-processing (resize, normalize) and the final
score threshold are the client's job.

## Repository layout

```
<backend>/config.json                    # per-backend spec
<backend>/pp_doclayout_v3_<backend>_<precision>.pte
labels.json                              # index -> class name, matching the `classes` output
```

## Output contract

A single **fixed-shape** method `forward`, declared in `config.json`. The RT-DETR box decode is
**baked into the graph** — outputs are ready to threshold:

```
in   [1, 3, 800, 800]     # RGB, normalized by the client to [0, 1] (x/255)
out  boxes   [300, 4]     # (x1, y1, x2, y2) in 800×800 model-input pixel space
     scores  [300]        # max-class sigmoid score per query
     classes [300]        # float class index per query (argmax)
```

Every dimension is static, so this model carries **no `get_model_schema` overrides** — a
client reads its signature straight from ExecuTorch's `MethodMeta`.

PP-DocLayoutV3 is a **DETR set-prediction** model → **no NMS**. All 300 queries are returned;
post-processing is just: keep rows with `score ≥ threshold`, scale boxes from the 800×800
input space to your image, and map `classes[i]` through `labels.json` (index → label).

### Classes (25)

`abstract, algorithm, aside_text, chart, content, display_formula, doc_title, figure_title,
footer, footer_image, footnote, formula_number, header, header_image, image, inline_formula,
number, paragraph_title, reference, reference_content, seal, table, text, vertical_text,
vision_footnote` — `labels.json` is the authoritative index→label map.

## Backends, sizes & latency (warm)

| backend | target | precision | size | latency |
|---|---|---|---|---|
| `xnnpack` | CPU | fp32 | 132 MB | ~2.0 s (Galaxy S24) |
| `coreml` | Apple ANE | fp16 | 91 MB | ~50 ms (Apple M-series ANE) |
| `vulkan` | Android GPU | fp16 (mixed-delegate) | **66 MB** | **~0.86 s (Galaxy S24)** |

> **Vulkan is the recommended Android backend** — ~2.4× faster than XNNPACK and half the size.
> It's mixed-delegate: most of RT-DETR runs fp16 on the GPU, while the box-head matmuls run on
> XNNPACK (they delegate as `addmm`→`linear`). XNNPACK stays fp32 because int8/int4
> quantization loses whole boxes on this model.

## Compatibility

If you intend to use these models outside of React Native ExecuTorch, make sure your runtime is
compatible with the **ExecuTorch** version used to export the `.pte` files. For more details, see
the compatibility note in the
[ExecuTorch GitHub repository](https://github.com/pytorch/executorch/blob/main/runtime/COMPATIBILITY.md).
If you work with React Native ExecuTorch, the library constants guarantee compatibility with the
runtime used behind the scenes.

These models were exported with ExecuTorch 1.3.1 and **no forward compatibility** is
guaranteed; older runtimes may not load them.
