# lfm2_5_text

## LFM2.5 2.6B instruction-tuned checkpoint

The `2_6b` size pins `LiquidAI/LFM2.5-2.6B` at revision
`654f9463ce32b05d0429d76fe1f580b27d4c1ac0`. It has 30 layers
(22 convolution, 8 attention), hidden size 2048, intermediate size 10752,
32 query heads, 8 KV heads, vocabulary 128000 and RoPE theta 10000000.
The generic `lfm2_5_1_2b` ExecuTorch class is reused with these parameters
and an explicit checkpoint; its built-in checkpoint auto-download is not used.

`_checkpoint.py` supports indexed safetensor shards and single-file weights,
and caches converted checkpoints by source repository and revision. Set
`LFM2_5_SOURCE_ROOT` to reuse a local snapshot in `<root>/2_6b/`.

Mobile artifacts limit context to **8192 tokens**, with **128-token** prefill
chunks, to keep cache allocations practical. All three backends expose float32
logits and an authored `get_model_schema` describing the dynamic prefill bound.
XNNPACK uses 8da4w linears (group size 32) and per-row int8 embeddings. MLX uses
int4 linears and portable int4 embeddings (group size 64). The latter avoids
TorchAO's optional native embedding quantizer, which is unavailable on macOS.
The decoder, attention, convolution and caches form one MLX partition; the
embedding lookup runs as a portable kernel.

```bash
uv run --project exporter_models/lfm2_5_text exporter export lfm2_5_text \
  --size 2_6b --backend xnnpack --output-dir out/2_6b/xnnpack
# Use the software-mansion-labs ExecuTorch MLX backend/runtime for RNE.
uv run --project exporter_models/lfm2_5_text exporter export lfm2_5_text \
  --size 2_6b --backend mlx --output-dir out/2_6b/mlx
```

Each backend output includes tokenizer assets, generation config and the
upstream license. The separate `chat_template.jinja` is embedded into
`tokenizer_config.json` for RNE. The Base checkpoint is intentionally omitted:
this export is for chat assistants. Existing 350M/1.2B recipes are unchanged.

### Monotonic thinking stripping

The upstream template strips thinking only from assistant messages before the
latest user message. Appending a new user message therefore rewrites old output
and fails the chat preprocessor's prefix check. Like the published Qwen3
template, our packaged template strips thinking from **every** assistant message
on every render. It keeps `<think>` in the generation prompt, so the model still
thinks before answering. We do not enable `preserve_thinking`.

The fix replaces the existing `tokenizer_config.json` on `v0.11.0`.

Validated with the real TypeScript chat preprocessor: two normal user turns and
a tool-call/result conversation followed by another user turn. Earlier thinking
is absent from every rendered prefix, and all incremental prefix checks pass.

### Native validation on macOS

`reference.py` (tested with transformers **5.14.1**) saves bf16 Hugging Face
logits. Run it separately from export/runtime checks to keep peak memory low
on a 16 GB machine. `validate.py` checks the actual `.pte` contract and constant
methods, compares the first prediction, generates up to EOS, and exercises
both single-token decode and the 128-token prefill bound. It records numeric
errors and generation timing in `validation.json`; this is a smoke test, not
a quality benchmark.

```bash
python exporter_models/lfm2_5_text/reference.py \
  --source /path/to/LFM2.5-2.6B --output reference.pt
python exporter_models/lfm2_5_text/validate.py \
  --model out/2_6b/mlx/lfm_2_5_2_6b_mlx_int4.pte --reference reference.pt \
  --native-runtime /path/to/matching/_portable_lib.cpython-310-darwin.so
# Repeat for the XNNPACK artifact with a runtime that includes XNNPACK.
```

The Mac runtime must register LLM and quantized embedding kernels as well as
the delegate. `--kernel-library` can load additional native registration
libraries when they are not linked into `_portable_lib`. A stock wheel's
Python MLX schema alone does not establish native runtime compatibility:
the installed 1.4.1 wheel failed execution, while the runtime built from the
RNE fork executed the same artifact correctly.

### Verified artifacts (2026-10-21)

Published under [`2_6b/` in the existing LFM2.5 repository](https://huggingface.co/software-mansion/react-native-executorch-lfm-2.5/tree/v0.11.0/2_6b).
The original 4K exports were uploaded at commit
`d5c844b222e78552765e3fba06b829f32f6095d5`; the current 8K exports are
described below.

Both instruction-tuned artifacts ran on an **Apple M1 Pro, 16 GB RAM**.
They matched the bf16 Hugging Face reference's first predicted token and
completed the chat answer with `Paris`. Both passed single-token decoding
and a 128-token prefill call with finite float32 `[1, 128000]` logits.

| Backend | File bytes | Delegate partitions | HF logit cosine (first step) |
| --- | ---: | ---: | ---: |
| XNNPACK 8da4w | 1,785,534,848 | 364 | 0.95924 |
| MLX int4 | 1,665,769,984 | 1 | 0.95474 |

The sibling `executorch-profiler` executed a 35-token prefill with event tracing:
XNNPACK delegate calls took 412.526 / 427.823 ms (96.42% of method time),
including delegated int4 fully connected operations; MLX took 158.255 /
158.354 ms (99.94%), with one portable embedding call. These individual
smoke-test timings are not a throughput or quality benchmark.

The matching Mac runtime was built from software-mansion-labs/executorch
commit `814c42d7d`, with MLX, XNNPACK and event tracing enabled. Its portable
Python module additionally loaded native LLM and quantized embedding kernel
registrations. The shipped iOS MLX library also contains the fork's CumMax
schema addition; the stock wheel's older compiled schema was the cause of
the initial MLX runtime failure.

## The XNNPACK 8da4w recipe changed

The first export of this model shipped a uniform `group_size 64` with no
embedding quantization. The sweep the old `exports/llms` directory ran (size /
throughput / perplexity / KL-divergence Pareto over fp16 and nine 8da4w
group-size combinations) picked different, per-size winners:

| size | sweep winner | what `_params.py` now sets |
|---|---|---|
| 350m | `8da4w_gs32_emb8_32` | `group_size 32`, `embedding_quantize "8,32"` |
| 1_2b | `8da4w_gs32_emb8_0` | `group_size 32`, `embedding_quantize "8,0"` |

Probing the published files confirms they predate this: neither carries a
quantized embedding table, matching the old uniform recipe rather than the
sweep. So `lfm_2_5_{350m,1_2b}_xnnpack_8da4w.pte` need a re-export to pick up
the recipe; the fp16 variants are unaffected.

The MLX backend quantizes to `4w` on its own schedule, unrelated to this sweep,
so it keeps its own `MLX_GROUP_SIZE`.

## Config

All four configs said `get_n_layers: null` where the `.pte` carry `16`; fixed.
Otherwise every published `.pte` here matches its config exactly -- the sequence
dim was already written as the real 2047 rather than 2048, unlike the six
families ported alongside it.

```bash
uv run --project exporter_models/lfm2_5_text python tools/probe_published_pte.py lfm2_5_text --size 350m
```

## 350M/1.2B checkpoint auto-download

The legacy 350M/1.2B sizes leave
`cfg.base.checkpoint` unset and lets `export_llama` resolve `model_class`
through its own `HUGGING_FACE_REPO_IDS`. That happens to be safe today -- the
table points `lfm2_5_350m` / `lfm2_5_1_2b` at `LiquidAI/LFM2.5-350M` and
`LiquidAI/LFM2.5-1.2B-Instruct`, which are the repos we want -- but it means an
executorch bump could silently repoint the weights. Worth converging on the
explicit-checkpoint shape the other models use; left alone here to keep this
change to the quantization recipe.

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

## LFM2.5 2.6B Vulkan and 8K context

The 2.6B configs use an 8192-token context, with the same 128-token prefill
bound. The upstream checkpoint supports 131072 positions; this does not change
RoPE scaling. Eight attention layers each cache K and V with eight 64-wide
heads: increasing context from 4096 to 8192 adds 128 MiB at float32 (64 MiB
at float16), before backend workspace. The 22 convolution caches stay fixed.

The Vulkan recipe uses 8da4w linears and int4 embeddings, both with group size
64. `force_fp16=True` selects float16 compute; this is separate from the
4-bit weight storage. Logits returned to the runner remain float32.
The forward graph lowers to one Vulkan delegate with no portable operators,
including embeddings, short convolutions, attention and mutable caches.
The initial 4K fp32 artifact's largest explicit 3D texture is 130 x 512 x 1;
large weights/caches use buffers, so the vocabulary does not create a giant
3D texture. The final 8K fp16 graph has explicit textures up to
2688 x 16384 x 128, requiring `maxImageDimension3D >= 16384`. The tested
Mali-G77 exposes 16384; GPUs with smaller limits need a different layout
or a partitioned export and have not been validated with this artifact.

On the connected Samsung Galaxy S20+ (SM-G986B, Exynos 990, Mali-G77, Android
13), using the Android libraries shipped with RNE, the initial 35-token chat
smoke test decoded at 13.7 tok/s with XNNPACK, 8.4 with Vulkan fp32 compute,
and 7.8 with Vulkan fp16 compute. Both Vulkan versions generated `Paris`.
The 406-token tool prompt prefills in 23.5 seconds with fp32 compute versus
11.8 seconds with fp16, so the final recipe uses fp16 despite its slightly
lower decode throughput. These are individual runs, not a thermal-controlled
benchmark. XNNPACK remains
the registry's Android default. The Vulkan artifact stays on Hugging Face
for manual testing, but is omitted from the registry and example because
it was slower on this device.

The 4K Vulkan fp32 export also completed the example's native tool round-trip:
406-token tool prompt, parsed `calculate(operation='multiply', a=17, b=23)`,
actual app tool result 391, then a correct final answer after the 438-token
follow-up. Prefill included multiple 128-token calls followed by single-token
decoding.

```bash
uv run --project exporter_models/lfm2_5_text exporter export lfm2_5_text \
  --size 2_6b --backend vulkan --output-dir out/2_6b/vulkan
```

`vulkan/validate_android.cpp` is a small standalone native smoke-test runner.
Build against the same vendored headers/libraries used by the app, link
`executorch`, `vulkan_executorch_backend`, and `xnnpack_executorch_backend`,
and deploy those libraries plus `libc++_shared.so` and `libfbjni.so` beside it.
The core library's Android SONAME is `libexecutorch_jni.so`; preserve that name
when deploying the file distributed as `libexecutorch.so`. Invoke with
`LD_LIBRARY_PATH` pointing at that directory:

```text
lfm_validate_android model.pte prompt.i64 output_prefix steps [chunk_size]
```

The prompt file holds little-endian int64 IDs rendered with the exported
chat template. The runner checks finite float32 `[1, 128000]` logits, writes
first-step logits and generated IDs, and reports prefill/decode timing plus
peak process RSS. RSS includes native runtime mappings and is not equivalent
to the model memory shown by the iOS app.

### Final 8K validation

All three artifacts passed finite-logit, dynamic-prefill, reference-first-token,
and chat-generation checks. The Mac runtime also exercised cache position
8191 with both XNNPACK and MLX. Long-prompt smoke results:

| Backend | Host | Prompt tokens | Prefill seconds |
| --- | --- | ---: | ---: |
| MLX | Apple M1 Pro, 16 GB | 8000 | 16.86 |
| XNNPACK | Apple M1 Pro, 16 GB | 7800 | 269.05 |
| XNNPACK | Samsung Galaxy S20+ | 8000 | 177.15 |
| Vulkan, fp16 compute | Samsung Galaxy S20+ | 8000 | 285.87 |

Each completed with the final answer `Paris`. The Mac XNNPACK prompt leaves
392 tokens for generation: an earlier 8000-token prompt consumed its remaining
192-token budget in reasoning. This is a context-capacity smoke test, not a
retrieval or long-context quality evaluation. Individual timings are sensitive
to host load and thermal state and must not be compared as a controlled
cross-device benchmark.

The final 8K Vulkan artifact also passed the full native tool round-trip through
the real example parser and calculation tool. On this phone XNNPACK decoded
at 15.0 tok/s for the short prompt versus Vulkan's 7.1; after 8000-token prefill
the corresponding runs measured 5.5 and 4.9 tok/s. Reports and artifact hashes
are published alongside the models in `2_6b/validation.json` and
`2_6b/export_manifest.json` at `v0.11.0`
(commit `59878269872c648d22f9e6815a4d4890d9728485`).
