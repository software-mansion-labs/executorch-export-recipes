# lfm2_5_colbert_350m notes

LFM2.5-ColBERT-350M late-interaction encoder (Linear 1024→128 per token +
per-token L2 norm → [B,S,128]), text-embedding, family lfm2.

## XNNPACK: dynamic seq, new-flow contract (2026-10-07, ET 1.4.1)

The static-512 specialization seen on executorch 1.3.1 is gone on the pinned 1.4.1
stack: the dynamic seq `{min:2,max:512}` survives lowering, so `xnnpack/config.json`
declares it and the `.pte` bakes `get_model_schema` (input_ids = attention_mask =
output seq). The runtime feeds PyLate's exact layout `[BOS, [Q]/[D], ...tokens]`;
BOS is NOT prepended in-graph (unlike the lfm2_5_embedding_350m exporters), because
the output is per token and must keep the input's seq length.

Quantization: torchao 8da4w (group 32) on every Linear except the short-conv ones,
which stay fp32 as in the embedding sibling, plus the word-embedding table
([64402, 1024]) quantized to int8 per row (`embedding_byte`), as in #1456. Host check
(Mac XNNPACK pybindings, inputs from `pylate.ColBERT.tokenize`, 24 multilingual
query/document pairs, reference = PyLate fp32; all arms 24/24 top-1):

| linears | embedding table | size | MaxSim corr | max dMaxSim | worst token cos |
|---|---|---|---|---|---|
| conv fp32 | fp32 | 570 MB | 0.983 | 0.28 | 0.967 |
| conv fp32 | int8 (shipped) | 373 MB | 0.982 | 0.31 | 0.964 |
| conv fp32 | int4 g32 | 348 MB | 0.981 | 0.30 | 0.961 |
| all 8da4w | fp32 | 426 MB | 0.973 | 0.48 | 0.957 |
| all 8da4w | int8 | 229 MB | 0.970 | 0.60 | 0.891 |
| all 8da4w | int4 g32 | 204 MB | 0.965 | 0.65 | 0.784 |

On device (RNE nlp app, same 24 pairs through `createColbertEmbedder`), the shipped
file matches its host run exactly (corr 1.00000, max dMaxSim 0.000) on a Galaxy S26
Ultra (17 ms/doc, 20 ms/query) and an iPhone 16 (44 / 57 ms). Quantizing the
short-conv linears did not scramble anything on the S26 (all-8da4w arm also matched
host); it only costs fidelity, per the table.

## MLX

The published `mlx/lfm_2_5_colbert_350m_mlx_int4.pte` (HF `f82677fc`) is
fork-serialized: `pte_inspector.py --mlx-instructions` with the
`@ms/separate-backends-1.4.1` schema shows 93 sane `QuantizedMatmulNode`s and
no `ArgPartition` garbage, and it carries `get_model_schema`. Not re-exported.
On an iPhone 16 it scores the 24 pairs 24/24, MaxSim corr 0.980 vs PyLate.

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
