# gemma4_omni notes

Gemma 4 E2B **omnimodal** (text + vision + audio). Three backends, from two
unrelated model ports:

| backend | port | model code | on device |
|---|---|---|---|
| `mlx` | ExecuTorch vendored | `_gemma4/` | yes |
| `xnnpack` | edge port | `_gemma4_edge/` | yes |
| `vulkan` | edge port | `_gemma4_edge/` | **NO** |

All three produce ONE `.pte` with four methods — `token_embedding`,
`text_decoder`, `vision_encoder`, `audio_encoder` — lowered together in a single
`to_edge_transform_and_lower` call with a PER-METHOD partitioner dict. On both
`mlx` and `vulkan` the audio encoder is routed to XNNPACK inside the otherwise
non-XNNPACK PTE.

Per-backend recipe detail is in `RECIPES.md`, the runner contract (calling
order, the splice, chat template, audio framing) in `MODEL_INTERFACE.md`, and
the optimisation history in `LEVERS.md` — copied verbatim from the former root
`gemma4/` package.

## ⚠ The vulkan backend requires a CHUNKING runner

`e2b/vulkan/config.json` and `vulkan/export.py` declare
`text_decoder_prefill_len = 128`, matching the multimodal Vulkan PTE published
on HF today. `vulkan/export.py` now carries a pointer to this section at that
constant, since its neighbours (`wrappers.py`, `RECIPES.md`) still call 768
"the fix".

**128 is correct and deliberate, not an under-sized bound.** The Vulkan
text_decoder garbles whenever a single `execute()` carries `seq_len >= 129` —
the cliff tracked as `project_gemma4_text_decoder_S128_cliff`
(`examples/models/llama/main_mm.cpp:157-163`). 128 is the largest value that
never trips it, so the graph bound and the runner's prefill chunk size are
deliberately the same number.

The encoders emit more than that — vision 280 soft tokens, audio up to 744 —
and that is fine **iff the runner chunks the prefill**:

| runner | chunks? | result |
|---|---|---|
| `llama_main_mm` | yes — `run_text_decoder_chunked(…, --prefill_chunk_size)`, default 128, multimodal path at `main_mm.cpp:2126`; KV cache shared across chunks via `input_pos` | multimodal works |
| stock `extension/llm/runner/multimodal_prefiller.cpp:215-228` | **no** — reads `seq_len` off the encoder output, one `execute()`, no bounds check | overruns → silent garbage |

React Native ExecuTorch is on the second path, so that is where the risk lives.
The failure mode is a property of the RUNNER, not of this `.pte`.

**Raising the bound is not the fix.** `TEXT_DECODER_PREFILL_LEN = 768` would let
one `execute()` reach S=768 and hit the very cliff 128 exists to avoid.
The TDS=768 variant is recorded in `RECIPES.md` as "the fix"; that attribution
predates the cliff finding, has never been run on a device, and should be
treated as unverified rather than as pending work. `RECIPES.md` and
`MODEL_INTERFACE.md` are verbatim upstream copies and still carry the older
explanation — this section supersedes them.

The dropped `export/validate.py` classified this configuration as an ERROR
(rule R5) and refused it without `--force`. That judgement stands; it lives here
now instead of in code.

## Two model ports coexist here, and they are not interchangeable

`_gemma4/` (MLX) and `_gemma4_edge/` (XNNPACK, Vulkan) are **independent ports**:
different class hierarchy (`Gemma4ForCausalLM` + `Gemma4Model` vs
`Gemma4EdgeForCausalLM`), different state-dict layout (a 100-entry remap vs a
prefix strip), zero shared files or symbols.

They also differ in how they get their encoders. `_gemma4/` pulls
`vision_tower` / `audio_tower` live off a loaded `transformers`
conditional-generation model. `_gemma4_edge/` carries from-scratch ports
(`model/vision.py`, `model/audio.py`) with the Vulkan graph-slimming flags —
though see the audio caveat below, where the edge tower is NOT the one used.

## `_gemma4_edge/` is byte-identical to gemma4's copy

Deliberate; keep them in sync. See `exporter_models/gemma4/NOTES.md` for why a
text-only subset was rejected.

```bash
diff -r exporter_models/gemma4/_gemma4_edge exporter_models/gemma4_omni/_gemma4_edge
```

### `sys.modules` keys are load-bearing

`gemma4_edge_omni` here, `gemma4_edge_text` under gemma4. `export/patches.py`
keeps its applied-set in module globals; two live copies sharing one key would
let patch P3's wrapping patch nest inside itself.

## What changed when this tree was vendored out of the root `gemma4/` package

Identical to the gemma4 model dir: absolute `gemma4.*` imports rewritten
relative; `presets.py` / `validate.py` / `__main__.py` dropped; `harness.py`
added (`HarnessRecipe` sources LLM constants from `config.json`, plus
`resolve_checkpoint`); `assets/e2b/` dropped as unread. Model logic is verbatim.

## Recipe decisions that are bisect results, not defaults

- **q/k/v/o_proj excluded from linear quantisation** (xnnpack). Bisected
  2026-04-23: quantising QKVO on top of the MLPs compounded drift far enough to
  flip the sampled token (10979 → 9366). They stay fp16.
- **Audio stays fp32** on both edge backends. An fp16 audio PTE gave min_cos
  0.85 on a row that flipped the first-token argmax 818 → 100, despite eager
  fp16 measuring 0.99996.
- **audio_encoder lowers on XNNPACK inside the Vulkan PTE.** The Vulkan
  `gemma_custom_sdpa` per-head + softcap shader measured cos 0.40 against eager.
  Every recipe since reroutes it; dropped rule R10 refused
  `audio_backend="vulkan"` outright.
- **The audio tower is the STOCK HF `Gemma4AudioTower`, not the edge port.**
  An earlier attempt with the edge tower produced uniform-noise logits on every
  modality and was reverted to the HF path, which is what ships. If an edge
  audio tower is ever reinstated, the trap is that its `bake_*`
  flags must be written to FOUR separate config objects, because each module
  captured its own reference.
- **Vision stays fp32 and unquantised on Vulkan** (the fp32 → fp16
  cast of bf16-trained vision params inflated output magnitude 5×). That is
  ~3.5 GB of the 4.37 GB PTE and the single largest open size lever.
- **PLE int2** on Vulkan saves ~580 MB but was measured to flip the multimodal
  first-token argmax (eager PLE int2 cos 0.895). `vulkan-mm` ships it anyway.
- **`audio_k_max = 62`, not 63.** An earlier XNNPACK recipe used 63, which
  exceeds the model's own
  `BLOCK_K_MAX` and admits a `k` the baked audio buffers were not sized for.
  Both edge recipes use 62 (29.76 s at 16 kHz, T_soft 744), which is what
  `MODEL_INTERFACE.md` and the runner assume.

## The xnnpack / vulkan configs are CORRECTED copies of the published contract

`e2b/xnnpack/config.json` and `e2b/vulkan/config.json` started as copies of
`software-mansion/react-native-executorch-gemma-4-multimodal`'s own
`e2b/<backend>/config.json`. The published files disagree with what the recipes
build, so they are vendored **corrected** — `skills/model-conventions` vendors a
local config precisely when the HF one is wrong. The same fixes are still owed
upstream.

- **All four methods now declare their I/O.** The published configs carry
  `{"inputs": [], "outputs": []}` for `token_embedding`, `text_decoder`,
  `vision_encoder` and `audio_encoder` — an explicit upstream TODO that left the
  harness with no contract to verify a `.pte` against. The blocks here are
  derived from the `torch.export` calls in `export/pipeline.py` and the wrapper
  docstrings in `export/wrappers.py`, NOT read off a built `.pte`; re-check them
  against `method_meta` on the next real export.
- **`image_placeholder_id`** was `262145`, the UPSTREAM `gemma3n` id that
  `MODEL_INTERFACE.md` says explicitly not to use; the local Gemma4 fork uses
  **258880**. The published config mixed lineages — its `audio_placeholder_id`
  (258881) was already the local-fork value. Since the config's constants are
  what `HarnessRecipe` bakes into the `.pte`, this shipped a model that told the
  runner to splice image tokens at a position the tokenizer never emits.
- **`get_max_seq_len`** was `4048` on xnnpack (equal to `get_max_context_len`)
  where the recipe's `text_decoder` bound is 1024, and the runner reads the key
  as the per-call prefill chunk size. Now 1024 / 768 respectively.
- **The audio keys were renamed** to the set `Recipe.constant_methods()` emits
  (`audio_block_samples`, `audio_block_k_{min,max}`, `audio_sample_count_*`,
  `audio_soft_tokens_max`). The xnnpack config used an older spelling
  (`audio_block_frames`, `audio_blocks_max: 63`, ...) that `check_constants`
  does not know, so those values passed unchecked — including the 63-vs-62
  discrepancy `RECIPES.md` had already resolved to 62.

There is no `KNOWN_CONFIG_DEVIATIONS` allowlist any more: `check_constants`
refuses any disagreement between config and recipe, because every declared
value is baked into the `.pte` verbatim.

## ⚠ No `schema.json` yet — the runtime cannot load these PTEs

Unlike gemma4 (text), this model ships no `schema.json` and bakes no
`get_model_schema` constant method, because **its interchange dtypes cannot be
expressed**. `token_embedding` emits `embeds` / `ple_tokens` as fp16,
`text_decoder` consumes them as fp16 and returns fp16 logits, and the Vulkan
`audio_encoder` deliberately casts its output to fp16 so the host splice into
the decoder's fp16 buffer is a clean memcpy.

The react-native-executorch runtime supports exactly `{uint8, int32, int64,
float32}`: the root `model.schema.json` DType enum has no `float16`, and
`dtypeFromScalarType` (`cpp/core/dtype.cpp`) throws "Unsupported ScalarType" on
`Half` — inside the `ModelHostObject` constructor, which walks every method. So
these PTEs fail to LOAD, schema or no schema.

Two ways forward, neither of them a config edit:

1. Land fp16 in the runtime's `DType` enum + the schema's, or
2. move the whole embeds/ple_tokens/logits interchange to fp32 — which doubles
   those buffers and is a numerics change nobody has measured on device.

Until one of them happens, the multimodal PTEs are exportable but not runnable
under the current runtime. The text model is unaffected: its only tensor
boundary is `tokens`/`input_pos` in, fp32 logits out.

## Shapes the recipes actually produce

Now also declared in both configs (see above); this section is the derivation.
Hidden 1536, layers 35, PLE dim 256, vocab 262144; values are the dynamic dims'
upper bounds.

**xnnpack** (`prefill_len` 1024, `text_decoder` bound 1024, `return_last_only`
true, logits fp16):

| method | inputs | outputs |
|---|---|---|
| `token_embedding` | `[1,1024]` i64 | `[1,1024,1536]` fp16, `[1,1024,35,256]` fp16 |
| `text_decoder` | `[1,1024,1536]`, `[1,1024,35,256]`, `[1024]` i64 | `[1,1,262144]` fp16 |
| `vision_encoder` | `[1,3,448,640]` fp32 | `[1,280,1536]` fp32 |
| `audio_encoder` | `[1,476160]` fp32, `[]` i64 | `[1,744,1536]` fp32 |

**vulkan** (`prefill_len` 2048, `text_decoder` bound 128, `return_last_only`
false, logits fp16):

| method | inputs | outputs |
|---|---|---|
| `token_embedding` | `[1,2048]` i64 | `[1,2048,1536]` fp16, `[1,2048,35,256]` fp16 |
| `text_decoder` | `[1,128,1536]`, `[1,128,35,256]`, `[128]` i64 | `[1,128,262144]` fp16 |
| `vision_encoder` | `[1,3,448,640]` fp32 | `[1,280,1536]` fp32 **+ `[1,280,35,256]` fp16** |
| `audio_encoder` | `[1,476160]` fp32, `[]` i64 | `[1,744,1536]` **fp16** |

Transcribed from `_gemma4_edge/export/wrappers.py` and the `torch.export` calls
in `pipeline.py:799-993`. **Not** read off a real `.pte`'s `method_meta` — do
that before trusting them.

## Per-method shape asymmetries

Not typos:

- **`vulkan` `vision_encoder` returns a 2-tuple** (`embeds` fp32 +
  `ple_tokens` fp16) while `xnnpack`'s returns `embeds` alone. Only the Vulkan
  path sets `emit_ple_tokens=True`, baking `ple_layer(pad_ids)` per layer as a
  constant — it drops 35× `aten.embedding` + 35× `dequantize_affine` from that
  method and keeps the int2 PLE tables in `text_decoder` only.
- **`audio_encoder` output is fp32 on xnnpack, fp16 on vulkan** — the Vulkan
  recipe appends `.to(fp16)` so the host splice into the fp16 embeds buffer is a
  plain memcpy.
- **`text_decoder` logits are `[1,1,vocab]` on xnnpack, `[1,S,vocab]` on
  vulkan** — `return_last_only` differs between the two recipes.
- **`token_embedding`'s bound (2048 on vulkan) differs from `text_decoder`'s**
  (128). The two dims are independent, which is the whole reason the multimodal
  PTE splits them: `token_embedding` must cover the full multimodal id sequence,
  while `text_decoder`'s dim is what the activation arena scales with.

Shapes are the dynamic dims' UPPER BOUNDS, per the `moonshine` convention.

## MLX export settings

`group_size=64` (the v3 recipe's CLI default is 32, but the published
multimodal artifact was produced at gs=64 — same on-device-tested setting as the
text `gemma4`). int4 weights; `max_seq_len=2048`. Baked into `mlx/export.py` to
match the shipped `.pte`. Note its config uses the UPSTREAM placeholder ids
(`image_placeholder_id: 262145`); the edge backends use the LOCAL Gemma4 fork's
(258880 / 258881). That difference is real — see `MODEL_INTERFACE.md`.

## `_gemma4/` self-containment (MLX)

Cross-imports were rewritten from `executorch.examples.models.gemma4.X` to
relative, so there is no `executorch.examples.models.gemma4` coupling. One
residual: `_gemma4/quant_utils.py:apply_embedding_quantization` lazily imports
`executorch.examples.models.llama...EmbeddingQuantHandler`, but that function is
only called on `export_gemma4.py`'s XNNPACK embedding-quant path, not the omni
MLX path — dead code, left verbatim rather than editing vendored model logic.

## Backend prerequisites

- **vulkan** needs patched `backends/vulkan/` kernels, not vendored here: the
  software-mansion-labs/executorch fork, branch `@mkopcins/gemma4`, at
  `0b7564fac554ab5fcb15bc12d03c607c21f11679`. Export and runtime must come
  from the same tree state; a mismatch gives gibberish, not an error.
  **That branch is ET 1.2.0a0, not the 1.3.1 the rest of this repo targets** —
  see gemma4/NOTES.md for why that is unresolved.
- **mlx** needs executorch built from the software-mansion-labs fork
  (`@ms/separate-backends`).

## Dependency stacks are unreconciled

Same situation as the gemma4 model dir: `pyproject.toml` pins the MLX stack
(python 3.10 / torch 2.14 nightly / transformers 5.14.1 / torchao 0.18 nightly)
while the edge recipes were validated on python 3.12.13 / torch 2.11.0 /
transformers 5.7.0 / torchao 0.15.0+git9338966da. Nobody has verified the edge
recipes on the pinned stack. Open work.

## Verification status

**mlx** — structural only.

**xnnpack / vulkan** — exported and shape-checked 2026-07-30 (host only; no
on-device run yet). xnnpack 815s / 4.687 GB, vulkan 404s / 4.371 GB — both
byte-sizes match the published artifacts exactly.

Every constant-method matches its config exactly on both backends (15 keys on
xnnpack, 17 on vulkan), including the deliberate `image_placeholder_id`
deviation: both `.pte`s bake the config's upstream 262145, not the local fork's
258880. `audio_placeholder_id` is 258881 (local fork) in both, so a runner sees
one id from each convention — see the deviation table above.

### The shapes the published configs omit

Both published configs declare `{"inputs": [], "outputs": []}` for all four
methods. These are the real ones, read off `method_meta`:

| method | xnnpack | vulkan |
|---|---|---|
| `token_embedding` in | `[1,1024]` i64 | `[1,2048]` i64 |
| `token_embedding` out | `[1,1024,1536]` f16, `[1,1024,35,256]` f16 | `[1,2048,…]` same shapes |
| `text_decoder` in | `[1,1024,1536]` f16, `[1,1024,35,256]` f16, `[1024]` i64 | **`[1,128,…]`** |
| `text_decoder` out | `[1,1,262144]` f16 (`return_last_only`) | `[1,128,262144]` f16 |
| `vision_encoder` | `[1,3,448,640]` f32 → `[1,280,1536]` f32 | → **2 outputs**: `[1,280,1536]` f32 + `[1,280,35,256]` f16 |
| `audio_encoder` | `[1,476160]` f32 + scalar i64 → `[1,744,1536]` f32 | → `[1,744,1536]` **f16** |

Two things worth not missing:

- **`vision_encoder` has a different arity per backend** — 1 output on xnnpack,
  2 on vulkan (it also emits PLE tokens). A runner written against one will not
  drive the other.
- **The vulkan `text_decoder` bound of 128 is correct, and is a runner
  contract.** Its own `vision_encoder` emits 280 and `audio_encoder` 744 into a
  method that accepts 128, which is fine *iff the runner chunks the prefill*.
  128 is the largest seq_len a single `execute()` may carry before the Vulkan
  attention cliff (`project_gemma4_text_decoder_S128_cliff`,
  `examples/models/llama/main_mm.cpp:157-163`), so the graph bound and the
  runner's chunk size are deliberately equal.
  `llama_main_mm` chunks (`run_text_decoder_chunked`, `--prefill_chunk_size`
  default 128, multimodal path at `main_mm.cpp:2126`) — multimodal works there.
  Stock `extension/llm/runner/multimodal_prefiller.cpp:215-228` does not: it
  reads seq_len off the encoder output and executes once, so RN-ET and anything
  else on that path overruns. Raising the bound is NOT the fix — it would let
  one `execute()` reach S=768 and hit the cliff 128 exists to avoid.

Vulkan delegation: `token_embedding` and `text_decoder` lower to **1 delegate
call** each; `vision_encoder` 3; `audio_encoder` **534**, because the audio
tower is deliberately rerouted to XNNPACK inside this Vulkan `.pte`.
