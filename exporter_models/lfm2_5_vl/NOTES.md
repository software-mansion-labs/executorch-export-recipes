# lfm2_5_vl notes

LFM2.5-VL vision-language model (vision_encoder + token_embedding + text_decoder),
sized `[450m, 1_6b]`, exported via the vendored `_lfm2_5_vl/` package (model code
from the executorch `examples/models/lfm2_5_vl` recipe; shared llama
transformer/kv-cache/sdpa infra kept as an import). One universal `export.py` per
backend reads `config["size"]` and maps it to `LiquidAI/LFM2.5-VL-{450M,1.6B}`.

## Why local config.json (not an HF pointer)

The published `react-native-executorch-lfm-2.5` repo does not ship a
`{size}/{backend}/config.json` for the VL sizes, so the multi-method VL contract
is hand-authored locally per `{size}/{backend}/`. 1.6B shapes scale with the text
hidden dim (2048 vs 450M's 1024): `token_embedding` out `[1,2048,dim]`,
`text_decoder` in `[1,2048,dim]`, `vision_encoder` out `[1,256,dim]`. The 1.6B
`vision_encoder` output dim is derived from the canonical config, not verified
against a shipped `.pte` — the harness will surface any mismatch at real export.

## Export specifics

The vendored `_lfm2_5_vl` already supports both sizes: `_resolve_params_path`
picks the bundled `config/lfm2_5_vl_{450m,1_6b}_config.json` from the model_dir
string. export.py passes `params_path` explicitly because upstream `export_all`
only resolves params in its CLI `main()`.

Producing the `.pte` needs an executorch source build (MLX backend for the `mlx/`
variants; XNNPACK otherwise).

## Core ML: not supported, and not attempted

This model has **no Core ML export and is not expected to get one.** Apple-silicon
acceleration here is MLX only (see `mlx/export.py`), which additionally requires the
software-mansion-labs ExecuTorch fork -- the PyPI wheel produces an MLX payload the
RNE runtime cannot load.

**Why Core ML is not on the table.** This is an autoregressive decoder driven by a
mutable KV cache. ExecuTorch's Core ML backend has no practical path for that: the
cache is a mutable buffer carried across calls, and the decode step needs a growing
sequence dim. Core ML's dynamic-shape support is the exact area where we already
have a confirmed, filed defect -- non-CPU compute paths reject shapes Core ML infers
as data-dependent (apple/coremltools#2825, see `docs/coremltools-2825-repro.md`).

**Status: a judgment call, not a measured result.** Nobody has run this model through
the Core ML backend and collected a failure. It is not attempted because the cost is
high and the prior is bad, not because it was tried and rejected. If Core ML gains
usable stateful-KV-cache support, this is worth revisiting.
