---
name: mlx-export
description: MLX backend knowledge for exporting ExecuTorch models — inline MLXPartitioner lowering, quantize_model_ (4w/8w), SwitchLinear MoE, the fp32-required-on-iOS gotcha, physical-device-only, transformers version pins.
---

# MLX Export

MLX is Apple's GPU backend. It is **physical-device-only** — models do not run
on the iOS simulator (the sim slice lacks the MLX delegate). Models lower
**inline** in their `mlx/export.py`.

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
- Some archs need a newer transformers than the repo default (e.g. the
  `openai_privacy_filter` MoE arch needs transformers >= 5.10.2; gemma4 pins
  5.7). Bump for the export, then restore.
- Interrupt+restart spikes RAM on MLX models (the backend cache isn't trimmed) —
  a runtime quirk, not an export defect.
