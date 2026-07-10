# lfm2_5_embedding_350m (xnnpack) notes

> **Update (react-native-executorch flow):** `xnnpack/export.py` and
> `mlx/export.py` now target the RNE `createTextEmbedder` contract: forward
> **prepends the `<|startoftext|>` CLS token (id 1)** and takes **raw, unpadded**
> tokens with an all-ones mask (the native runtime tokenizer emits raw tokens),
> and the program ships a **`get_dynamic_dims_forward` companion** advertising the
> `[1, max_tokens]` sequence range — so the metadata-static `.pte` still accepts
> variable-length input (this supersedes the "static seq 512" workaround below).
> Config seq is `511` (forward adds the CLS token → 512 total). Verified: fp32
> cosine 0.9999 vs the sentence-transformers reference; xnnpack 8da4w verified on
> a physical Android device. MLX int4 exports but is **not** yet verified on iOS.

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

The `mlx/` backend (selected with `--backend mlx`) fetches its config from the HF
pointer and is structural-only — the MLX backend needs an ET source build, not
verified here.
