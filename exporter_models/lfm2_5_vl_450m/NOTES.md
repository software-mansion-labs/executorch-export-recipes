# lfm2_5_vl_450m notes

LFM2.5-VL-450M vision-language model (image encoder + text decoder), exported
via the vendored `_lfm2_5_vl/` package (model code from the executorch
`examples/models/lfm2_5_vl` recipe; shared llama transformer/kv-cache/sdpa infra
kept as an import).

## Why a local config.json (not a model.json HF pointer)

The **450M** size is NOT published on HuggingFace — only
`software-mansion/react-native-executorch-lfm2.5-VL-1.6B` exists (with a
`quantized/config.json`). There is no `react-native-executorch-lfm2.5-VL-450M`
repo to point a `model.json` at, so the multi-method VL contract
(vision_encoder + token_embedding + text_decoder) is hand-authored locally.

If/when a 450M repo is published with a canonical spec config, switch both
backend subdirs to a `model.json` pointer and drop the local configs.

## Export specifics

`export.py` resolves the 450M params via the vendored
`_resolve_params_path("LiquidAI/LFM2.5-VL-450M", None)` → bundled
`_lfm2_5_vl/config/lfm2_5_vl_450m_config.json` (text dim 1024), NOT the 1.6B
default (dim 2048) — the upstream `export_all` only resolves params in its CLI
`main()`, so export.py passes `params_path` explicitly.

The `mlx/` backend (`--backend mlx`) is structural-only (MLX backend needs an ET
source build).
