---
name: mlx-export
description: MLX backend knowledge for exporting ExecuTorch models — inline MLXPartitioner lowering, quantize_model_ (4w/8w), SwitchLinear MoE, the fp32-required-on-iOS gotcha, physical-device-only, transformers version pins.
---

# MLX Export

MLX is Apple's GPU backend. It is **physical-device-only** — models do not run
on the iOS simulator (the sim slice lacks the MLX delegate). Models lower
**inline** in their `mlx/export.py`.

For failures follow [recover-export](../../recover-export/SKILL.md), especially
exporter/runtime schema compatibility. For quality and speed comparisons use
[validate-and-optimize](../../validate-and-optimize/SKILL.md); Mac execution
alone does not validate the iOS delegate.

## Recipe

```python
from executorch.backends.mlx import MLXPartitioner          # or .partitioner import MLXPartitioner
from executorch.backends.mlx.passes import get_default_passes
from executorch.extension.llm.export.quantize import quantize_model_
from executorch.exir import to_edge_transform_and_lower

quantize_model_(model, qlinear_config="4w", qlinear_group_size=64,
                qembedding_config="4w", qembedding_group_size=64)
ep = torch.export.export(wrapper, example_inputs, strict=...)
edge = to_edge_transform_and_lower({"forward": ep},
    transform_passes=get_default_passes(), partitioner=[MLXPartitioner()])
et = edge.to_executorch()
```

For LLMs the `export_llama` / `export_llm` path is also used (LFM2.5 text:
`cfg.backend.mlx.enabled = True`, `cfg.quantization.qmode = "4w"`).

## Quantization

- `4w` = int4 weights, `8w` = int8 weights; group size 64 is the common default.
- **Quantize the embedding too** (`qembedding_config`) — for large-vocab models
  the unquantized embedding is the dominant size leak.

## SwitchLinear / MoE

MoE models (e.g. privacy filter) replace the stock data-dependent experts loop
with a static form backed by `executorch.backends.mlx.llm.switch.SwitchLinear`
(int-packed `mlx::gather_qmm`) + `pack_all_switch_linears(model)`. Gather expert
biases via `F.embedding(router_indices, stacked_bias)` (NOT `tensor[idx]`, which
lowers to a take_along_axis MLX can't broadcast). Build dense routing with
`one_hot`+sum, not scatter.

## fp32 is REQUIRED on the iOS MLX delegate for the MoE path

bf16 corrupts the MoE `gather_qmm` path on the iOS delegate (every token comes
back mislabeled) while being clean on Mac MLX — so Mac verification is
misleading here. Export MoE models with **fp32** compute dtype. It is nearly
free since the expert weights are int-quantized regardless; only the small
unquantized tensors (norms, biases, scales, head) change size. Plain (non-MoE)
decoders tolerate bf16.

## Environment / pins

- The MLX backend + delegate must be installed; export and on-device run need a
  physical Apple device.
- Some architectures need different transformers versions (e.g. privacy
  filter and Gemma). Use the owning model's verified pin or update that model's
  environment with validation evidence; do not mutate another model's env.
- Interrupt+restart spikes RAM on MLX models (the backend cache isn't trimmed) —
  a runtime quirk, not an export defect.

## Computer-vision models (#1318): exported & verified, but DON'T SHIP

The 8 CV models with a CoreML variant (efficientnet_v2_s, ssdlite320, the four
style_transfer presets, rfdetr_nano detector + segmentation) were given MLX
exports and verified on a physical device (iPhone SE 3rd gen). **All run
correctly, but every one is slower than its CoreML variant** — CoreML lowers to
the ANE, purpose-built for these conv nets, while MLX runs on the Metal GPU
(MLX's real win is LLMs, not CV). Benchmarked RF-DETR (the closest case, same
image, 30 iters / 5 warmup): **CoreML INT8 avg 51 ms vs MLX FP16 avg 68 ms**
(~33% slower, worse tail). fp16 (below) narrowed but never closed the gap. These
`mlx/` dirs are kept as a **reference recipe**; the `.pte` were NOT uploaded to
HF and MLX is NOT wired into the app's `models.ts`.

Recipe details worth keeping (the `export.py` here already encode them):

- **fp16 = `model.half()` + an fp32-I/O wrapper.** MLX has no CoreML-style
  compile-time `compute_precision`; precision comes from the torch model dtype.
  Wrap the model so `forward` casts fp32 input → fp16, runs, casts float outputs
  → fp32 (mirrors CoreML FLOAT16 and preserves the app's fp32 I/O contract). See
  `_Fp16IO` in the CV `export.py`. Halves file size.
- **ssdlite can't go fp16** — torchvision's SSD anchor generator does an
  fp16/fp32 `torch.cat` under `.half()` (dtype mismatch). CoreML sidesteps this
  by lowering an fp32 graph to fp16 at compile time; MLX has no equivalent. It
  stays fp32.
- **DETR (rfdetr det/seg) needs a fork fix.** The MLX builder crashes on a
  `getitem` off a boundary-partitioned multi-output op (`native_layer_norm`,
  which DETR has dozens of): `'Slot' object is not subscriptable`. Guard
  `_getitem_handler` in `backends/mlx/ops.py` to forward a single boundary-input
  Slot directly. Also split `max(dim=-1)` into `amax`+`argmax` (the multi-output
  `aten.max.dim` getitem hits the same builder path). This also blocks the
  pre-existing `rfdetr_keypoint/mlx` (which has a separate meta-tensor issue).
- **InstanceSegmenter signature.** The runtime validates a 4-output, batch-free
  `forward`: `boxes[N,4]`, `scores[N]`, `labels[N]`, `masks[N,MH,MW]` — return
  the four separately and index `[0]` to drop the batch dim (the detector wrapper
  already does the batch-free part).
