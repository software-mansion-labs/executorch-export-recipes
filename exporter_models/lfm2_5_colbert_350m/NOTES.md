# lfm2_5_colbert_350m (xnnpack) notes

LFM2.5-ColBERT-350M late-interaction encoder (Linear 1024→128 per token +
per-token L2 norm → [B,S,128]), text-embedding, family lfm2. Exported from the
upstream publishing recipe `examples/models/lfm2_5_retrieval/produce_colbert_pte.py`
(full 8da4w, group_size 32).

## Local config (static seq 512) — deliberate deviation from the published config

Same executorch-1.3.1 dynamic-seq specialization as the embedding model: the
published HF config declares `{min:2, max:512}` on the seq dim (inputs and the
[B,S,128] output), but XNNPACK lowering on this build bakes it to static
`[1,512]` / `[1,512,128]`. Root-caused (dynamic dim survives `torch.export`,
specialized at lowering; recipe-faithful and `_skip_dim_order` both reproduce
the static result). The published dynamic `.pte` predates this build.

This dir ships a LOCAL `config.json` (no `model.json`) pinned to the static
`[1,512]` / `[1,512,128]` contract — verified `io_ok` + `parity_ok`. Revert to
the HF-pointer `model.json` once an ET build preserves the dynamic seq through
XNNPACK lowering. The `_mlx` sibling keeps the HF pointer (structural-only).
