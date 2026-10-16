# lfm2_5_embedding_350m

> **Update (react-native-executorch flow):** `xnnpack/export.py` and
> `mlx/export.py` now target the RNE `createTextEmbedder` contract: forward
> **prepends the `<|startoftext|>` CLS token (id 1)** and takes **raw, unpadded**
> tokens with an all-ones mask (the native runtime tokenizer emits raw tokens),
> and the program ships a **`get_model_schema` constant method** advertising the
> `[1, max_tokens]` sequence range — so the metadata-static `.pte` still accepts
> variable-length input (this supersedes the "static seq 512" workaround below).
> Config seq is `511` (forward adds the CLS token → 512 total). Verified: fp32
> cosine 0.9999 vs the sentence-transformers reference; xnnpack 8da4w verified on
> a physical Android device.

## MLX must be exported with the runtime-locked fork (Error::InvalidProgram)

The MLX delegate serializes its graph as a FlatBuffer union, and the union member
order *is* the wire format. The fork the shipped `MLXBackend` runtime is built
from (software-mansion-labs/executorch @ `@ms/separate-backends-1.4.1`) carries
`CumMaxNode` at union value 112, which upstream does not have, so every op from
there on is shifted by one against the PyPI `executorch` wheel this model's
`pyproject.toml` pins for xnnpack/coreml. Both sides write schema version `"1"`,
so nothing catches the skew.

The `.pte` published at HF tag `v0.10.0` was built with the wheel: its 92
quantized matmuls carry union value 121, which the runtime reads as
`ArgPartitionNode`, and loading dies in `MLXBackend::init` with
`Error::InvalidProgram` ("Load method 'forward' failed"). Found on an iPhone 17
in the v0.10.0 estate sweep — the only failure out of 191 variants.

`mlx/export.py` now calls `exporter.backends.require_mlx_runtime_fork()`, which
refuses to export unless union value 112 is `CumMaxNode`.

To export MLX, overlay the fork's AOT package into this model's env (a full
`uv` source build of the fork is not resolvable here):

```
git -C <executorch-fork> archive @ms/separate-backends-1.4.1 backends/mlx | tar -x -C /tmp/forkmlx
B=exporter_models/lfm2_5_embedding_350m/.venv/lib/python3.10/site-packages/executorch/backends
mv $B/mlx $B/mlx.wheel.bak && cp -R /tmp/forkmlx/backends/mlx $B/mlx
(cd $B/mlx/serialization && python generate.py)   # needs flatc on PATH
uv run --no-sync --project exporter_models/lfm2_5_embedding_350m \
  exporter export lfm2_5_embedding_350m --backend mlx --output-dir <dir>
```

Verify the result with the fork's inspector: the 92 quantized linears must
decode as `QuantizedMatmulNode` (group_size 64, bits 4, affine), not
`ArgPartitionNode`.

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

## Vulkan: exporter kept, nothing published

`vulkan/export.py` and `vulkan/config.json` produce two working, verified
artifacts (fp16 and int8), but no `.pte` is published. Both correctness gates
passed -- nearest-neighbour ranking is identical to fp32 -- so this is purely a
performance verdict. `vulkan` stays in `model.json` backends so
`--backend vulkan` runs and the finding can be re-tested.

Measured 2026-09-02 on Adreno 840 (S26 Ultra) and Mali-G76 (S10+), 511 tokens:

| build | Adreno 840 | Mali-G76 | size | cosine vs fp32 |
|---|---|---|---|---|
| Vulkan int8 | **330.9 ms** (1.09x faster) | **28,931 ms** (5.4x slower) | 443 MB | 0.999648 |
| XNNPACK 8da4w (shipped) | 362.3 ms | 5,385 ms | 575 MB | - |
| Vulkan fp16 | 466.3 ms (1.29x slower) | 5,535 ms (1.03x slower) | 676 MB | 0.999985 |

The two variants fail on opposite devices, so neither is shippable as a default:

- int8 is the only build that beats XNNPACK anywhere (Adreno, 1.09x) and is 5.4x
  slower on Mali. This is the same regression that kept int8 out as the mpnet
  default (5-8x there). Two models now, so treat "ET Vulkan int8 tiled kernels
  are Adreno-tuned and pathological on Mali" as a rule rather than a quirk.
- fp16 is safe on Mali (1.03x, parity) but never wins anywhere.

Do not shorthand this as "Vulkan loses on Mali": only **int8** does. fp16 on Mali
is fine, which matters because MiniLM fp16 won 2.4x there. Test both precisions
on both GPUs before any verdict.
