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
XNNPACK lowering. The `mlx/` backend (`--backend mlx`) fetches from the HF pointer
and is structural-only.

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

`coreml` is therefore **not** in this model's `model.json` backends. The exporter is
kept so the finding can be re-tested.
