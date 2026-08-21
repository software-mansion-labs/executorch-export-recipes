# lfm2_5_embedding_350m (xnnpack) notes

> **Update (react-native-executorch flow):** `xnnpack/export.py` and
> `mlx/export.py` now target the RNE `createTextEmbedder` contract: forward
> **prepends the `<|startoftext|>` CLS token (id 1)** and takes **raw, unpadded**
> tokens with an all-ones mask (the native runtime tokenizer emits raw tokens),
> and the program ships a **`get_model_schema` constant method** advertising the
> `[1, max_tokens]` sequence range — so the metadata-static `.pte` still accepts
> variable-length input (this supersedes the "static seq 512" workaround below).
> Config seq is `511` (forward adds the CLS token → 512 total). Verified: fp32
> cosine 0.9999 vs the sentence-transformers reference; xnnpack 8da4w verified on
> a physical Android device. MLX int4 exports but is **not** yet verified on iOS.

## transformers >= 5.x short-conv `seq_idx` (export bind failure)

Under transformers 5.x the Lfm2 decoder layer calls the short-conv with a
`seq_idx` kwarg (packed-sample conv-state reset). The model's remote
`_noncausal_shortconv_forward` override predates that arg, so `torch.export`
fails with *"failed to bind arguments … `_noncausal_shortconv_forward`"*. Both
backend exports now call `patch_shortconv_static_length(inner)` before export,
which reinstalls the non-causal short-conv with a `seq_idx=None, **kwargs`
signature (numerics unchanged — seq_idx is always None here). Verified on
executorch 1.3.1 / transformers 5.14.1: xnnpack re-export is byte-identical in
size to the prior verified build and eager parity holds (L2-normed [1,1024]).

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

## Core ML: exporter exists, but the artifact does not load

`coreml/export.py` was added 2026-08-20 and **converts successfully** (fp16, ~710 MB),
but the resulting `.pte` fails to load in the ExecuTorch runtime:

```
[tensor_parser_portable.cpp:192] getTensorDataPtr() failed: 0x23
[method.cpp:606] Failed parsing tensor at index 0: 0x23
```

Note what this is **not**: it is not the dynamic-shape crash that disables the other
Core ML text models (apple/coremltools#2825). This fails at load, in the *portable*
tensor parser, meaning ops the Core ML partitioner did not claim kept their constants
in the ExecuTorch program and that data is unreadable -- plausibly the artifact size
or a constant-segment issue. The dynamic-shape question is therefore still **open**
for this model: it never executed, so it never got far enough to hit it.

The Core ML variant also differs from the MLX/XNNPACK ones in two deliberate ways:
the model is loaded in **float32** (Core ML has no bfloat16 input dtype) and
`quantize_model_` is **not** applied (it packs weights for the MLX runtime), so the
Core ML build is unquantized.

**Re-tested 2026-08-20 with the software-mansion-labs fork toolchain** (rebuilt and
loaded under the fork runtime): identical `0x23` failure. So this is not a
PyPI-wheel artifact, unlike the YOLO fp16 MLX case which the fork did fix.

No Core ML `.pte` is published for this model. The exporter and its
`coreml/config.json` are kept, and `coreml` stays in `model.json` backends, so
`--backend coreml` actually runs and the finding can be re-tested. (It was
previously dropped from that list, which made the exporter unreachable and this
paragraph impossible to act on.)
