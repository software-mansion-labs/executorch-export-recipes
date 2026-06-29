# lfm2_5_embedding_350m (xnnpack) notes

LFM2.5-Embedding-350M dense bi-encoder (CLS pool + L2 norm → [B,1024]),
text-embedding, family lfm2. Exported from the upstream publishing recipe
`examples/models/lfm2_5_retrieval/produce_embedding_pte.py` (8da4w with the
short-conv linears left in fp32 — `_quantize_8da4w_skip_conv`; quantizing the
conv linears scrambles embeddings on ARM XNNPACK).

## Local config (static seq 512) — deliberate deviation from the published config

The published HF config declares a DYNAMIC sequence dim `{min:2, max:512}`.
Under **executorch 1.3.1**, XNNPACK `to_edge_transform_and_lower` specializes
the dynamic seq dim to its max, so the re-exported `.pte` reports a STATIC
`[1, 512]` regardless of recipe fidelity (verified: the dynamic dim survives
`torch.export` as a symbolic `s11`, but is baked during lowering; `_skip_dim_order`
does not change this; running the exact producer recipe reproduces it). The
published `{2..512}` `.pte` was produced on an older executorch build whose
XNNPACK path preserved the dynamic dim.

So this dir ships a LOCAL `config.json` (no `model.json` pointer) pinned to the
static `[1,512]` contract this executorch build actually produces — verified
`io_ok` + `parity_ok`. When an executorch build that preserves the dynamic seq
through XNNPACK lowering is used, switch back to the HF-pointer `model.json`
(`{"repo":"software-mansion/react-native-executorch-lfm2.5-embedding-350m","config":"xnnpack/config.json"}`)
and drop this local config.

The `_mlx` sibling dir keeps the HF pointer (structural-only — MLX backend needs
an ET source build, not verified here).
