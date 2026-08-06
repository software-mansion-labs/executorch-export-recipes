# gemma4 notes

Gemma 4 E2B **text decoder**. Three backends, from two unrelated model ports:

| backend | port | model code | on device |
|---|---|---|---|
| `mlx` | ExecuTorch vendored | `_text_decoder/` | yes |
| `xnnpack` | edge port | `_gemma4_edge/` | yes |
| `vulkan` | edge port | `_gemma4_edge/` | yes |

All three produce a single-method PTE whose method is named `forward` —
`extension/llm/runner/text_decoder_runner.h` hardcodes `load_method("forward")`
and a single-method PTE named anything else aborts at load.

Per-backend recipe detail lives in `RECIPES.md`, the runtime contract in
`MODEL_INTERFACE.md`, and the confirmed/falsified optimisation history in
`LEVERS.md` — all three copied verbatim from the former root `gemma4/` package.

## Two model ports coexist here, and they are not interchangeable

`_text_decoder/` (MLX) and `_gemma4_edge/` (XNNPACK, Vulkan) are **independent
ports of Gemma 4**, not two versions of one:

| | `_gemma4_edge/model/` | `_text_decoder/` |
|---|---|---|
| Top class | `Gemma4EdgeForCausalLM` | `Gemma4ForCausalLM` + `Gemma4Model` |
| Layout | one 1830-line `text.py` | 8 files, one class each |
| KV sharing | inline, via `num_kv_shared_layers` | `Gemma4SelfDecoder` / `Gemma4CrossDecoder` |
| State dict | HF names, prefix-strip load | remapped through a 100-entry dict |

They share no file and no symbol. A checkpoint conversion written for one does
not apply to the other. Do not try to unify them without a reason.

## `_gemma4_edge/` is byte-identical to gemma4_omni's copy

Deliberate. `export/pipeline.py` and `export/wrappers.py` import `model.audio`,
`model.vision` and `model.mel_frontend` at MODULE scope, so a text-only subset
would mean editing `pipeline.py` — the one file whose nine ordering constraints
must not break — and would diverge the two copies. Identical trees stay
diffable:

```bash
diff -r exporter_models/gemma4/_gemma4_edge exporter_models/gemma4_omni/_gemma4_edge
```

The encoder modules are therefore dead weight in this model dir. Keep the two
copies in sync when changing either — the `diff -r` above must stay empty, and
it is the only thing standing between "duplicated on purpose" and "forked by
accident" (it had already drifted once, in comments only).

**Consequence for dependencies:** `wrappers.py` imports `model.mel_frontend`,
which imports `transformers.audio_utils` at module scope. The text export
therefore needs `transformers` even though it exports no audio.

## What changed when this tree was vendored out of the root `gemma4/` package

- Absolute `gemma4.model.*` / `gemma4.export.*` imports were rewritten
  relative, so the tree is importable under any `sys.modules` key. Model logic
  is otherwise verbatim.
- `export/presets.py` was dropped — each `<backend>/export.py` now declares its
  recipe as module-level constants, matching `mlx/export.py`'s house style.
- `export/validate.py` was dropped — its rules are prose, below.
- `export/__main__.py` was dropped — the CLI is `exporter export`.
- `export/harness.py` was ADDED: `HarnessRecipe` (LLM constants come from
  `config.json`, not from the recipe) and `resolve_checkpoint`.
- `assets/e2b/` was dropped. No code path read it; the pipeline loads
  `config.json` out of the checkpoint directory.

### `sys.modules` keys are load-bearing

`gemma4_edge_text` here, `gemma4_edge_omni` under gemma4_omni. `export/patches.py`
keeps its applied-set and its saved originals in module globals; two live copies
sharing one key would let patch P3's wrapping patch nest inside itself — the
exact failure its idempotence guard exists to prevent.

## The xnnpack / vulkan configs are CORRECTED copies of the published contract

`e2b/xnnpack/config.json` and `e2b/vulkan/config.json` started as copies of
`software-mansion/react-native-executorch-gemma-4`'s own
`e2b/<backend>/config.json`, but the published files disagree with what the
recipes actually build, so they are vendored **corrected** — which is what
`skills/model-conventions` asks for ("vendor a local config only when ... the
HF config is wrong"). The published files still need the same fixes upstream.

What was corrected, and why each one matters rather than being cosmetic:

| | published | vendored here |
|---|---|---|
| `forward` inputs | `tokens` only | `tokens` + `input_pos` |
| `forward` seq dim | `{"min": 1, "max": 2048}` | 1024 (xnnpack) / 128 (vulkan) |
| logits dtype | float32 | float32 (the graph now emits fp32 too) |
| `get_max_seq_len` | 4048 (xnnpack) | 1024 |
| `get_max_context_len` | 4048 / 2048 ✓ | unchanged |

- **`input_pos` was missing** from the published `forward`, which traces as
  `(input_ids, input_pos)` with `use_kv_cache` true — a KV-cache decode loop
  cannot run without it.
- **The seq dim** was capped at 2048 where the graphs are bounded at 1024 and
  128. `validateSpec` in the runtime rejects a declared bound above the
  compiled one, so 2048 was not merely inaccurate.
- **`get_max_seq_len` = 4048 on xnnpack** equalled `get_max_context_len`, while
  the runner reads it as the PER-CALL PREFILL CHUNK SIZE (rule R8). A runner
  chunking prefill at 4048 overruns this graph. It is now the graph's own
  prefill bound.

Every declared constant is written into the `.pte` verbatim by
`HarnessRecipe.constant_methods()`, so `harness.check_constants` refuses any
disagreement with the recipe outright. There is no deviation allowlist: a
tolerated mismatch ships an artifact whose metadata lies about its own graph.

## `get_model_schema` — the runtime I/O contract

`schema.json` at the model root is the spec the react-native-executorch runtime
actually reads. Each `<backend>/export.py` runs it through
`exporter.schema.load_and_validate_schema` (templated with that backend's
`{seq_max}` and the config's `{vocab_size}`, validated inline against the repo
root `model.schema.json`) and `pipeline.export` bakes the JSON into the `.pte`
as the `get_model_schema` constant method.

It carries what `config.json` cannot express to the runtime:

- **Range dims.** Without the companion, `ModelHostObject` falls back to
  `methodSpecFromMetadata`, where every dimension is the static compiled upper
  bound — a decode call at S=1 against an S<=1024 graph is then rejected.
- **An equality constraint** tying `tokens` dim 1, `input_pos` dim 0 and the
  logits' seq dim together, which the runtime checks before every execution.

**Only `float32` / `int64` / `int32` / `uint8` are expressible** — the schema's
DType enum and the runtime's `dtypeFromScalarType` agree on exactly those four,
and `Half` throws at load. That is why `LOGITS_DTYPE` is fp32 here (see the
note in `xnnpack/export.py`) and why gemma4_omni ships no `schema.json` yet.

## Rules the dropped validator used to enforce

- **R8** — `get_max_seq_len` is the runner's PER-CALL PREFILL CHUNK SIZE, not
  the context limit. `get_max_context_len` is the context limit. Both configs
  set `get_max_seq_len` to the graph's seq bound deliberately, so the runner
  chunks at exactly what the graph accepts; the cost is that one prefill call is
  capped at that value.
- **R1–R3** — the Vulkan SDPA source transform REJECTS ring KV,
  `use_dynamic_kv_slice` and `use_bshd_kv=False` by raising. Everything the
  XNNPACK recipe turns on for host-op reduction is unavailable on Vulkan; the
  fused custom op does that work in the shader.
- **R11** — `use_bshd_kv=False` costs ~280 extra transposes. No shipped recipe
  uses it.

## Backend prerequisites

- **vulkan** depends on patched `backends/vulkan/` kernels — the softmax
  max-subtraction fix, the gelu/tanh fp16 clamp (both to −10/10),
  `GemmaSDPA.cpp` implementing the two `et_vk::` ops, the `op_registry.py`
  entries that let the partitioner claim them, and several `resize_fn` fixes.
  They are NOT vendored here: they live in the software-mansion-labs/executorch
  fork on branch `@ms/separate-backends` — **the same branch mlx already uses**,
  pinned in `pyproject.toml`. Verified present at
  `2daed17d827b7d7cc2f61d44dc20794c81d8da94` (ET 1.3.1).

  A PTE exported against one set of shaders and run against another produces
  gibberish at runtime, not an error — check this first when on-device output
  is nonsense.

  The kernels originated on the fork's `@mkopcins/gemma4` branch, which is ET
  **1.2.0a0**, and were **adapted rather than copied** onto 1.3.1: the shaders
  there call `load_{q_projected,k_cache}_tile_with_checks` with an extra `D`
  argument and `#define Q_LAYOUT/K_LAYOUT DHSB`, matching 1.3.1's tile-load
  helpers. That is a semantic change to the attention shaders, so the
  on-device validation recorded for these recipes was done against the 1.2
  shaders, not these — re-run the on-device smoke before trusting the numbers.
- **mlx** needs executorch built from the same fork and the same branch,
  `@ms/separate-backends` (ET 1.3.1), version-locked to the MLX runtime.

## Export settings are the on-device-tested invocations

**mlx** — the recipe is `export_mlx_standalone.py`, whose `export()` calls
`export_gemma4_e2b_mlx` with BAKED-IN, on-device-tested settings ("int4 /
group_size 64, bf16, compile-friendly"): `dtype="bf16"`, `qlinear="4w"`,
`qembedding="4w"`, `group_size=64`, `compile_friendly=True`, `max_seq_len=2048`.
These are NOT the function-signature defaults (fp32 / gs32 /
`compile_friendly=False`) — they are what the standalone passes, and what
produced the published `gemma4_e2b_mlx_int4.pte`.

**xnnpack** — the on-device-tested values, with two deliberate deviations
carried over from the recipe package: `ple_embedding_bits=4` where the shipped
text bundle used int2 (int2 saves ~580 MB but flipped the multimodal first-token
argmax, and that int2 PTE was flagged as still owing an on-device chat smoke),
and `return_last_only=False` (the source recipe's docstring claimed True while
its code said False — the code is what shipped).

**vulkan** — the on-device-tested values verbatim. Two that look wrong and are
not:
`logits_dtype` stays fp32 while the rest of the model is fp16, and
`prefill_len=128` so runners must chunk prefill at ≤128 while the KV cache still
serves 2048 positions.

## Dependency stacks are unreconciled

`pyproject.toml` pins the MLX stack — python 3.10, torch 2.14 nightly,
transformers 5.14.1, torchao 0.18 nightly. The XNNPACK and Vulkan recipes were
validated on a **different** stack: python 3.12.13, torch 2.11.0, transformers
5.7.0 (must provide `models/gemma4`), torchao 0.15.0+git9338966da.

One `pyproject.toml` can pin only one. The MLX pin is kept because all three
backends need an ExecuTorch source build anyway (MLX the `@ms/separate-backends`
fork, Vulkan the patched kernels), so no `uv.lock` is produced either way and
the pin is documentation rather than machinery. **Nobody has verified the edge
recipes on the pinned stack.** Reconciling them is open work.

## Verification status

**mlx** — structural only: spec resolution, export-script discovery and plan.

**xnnpack / vulkan** — exported and shape-checked 2026-07-30 (host only; no
on-device run yet). Both built from `$GEMMA4_CHECKPOINT=.../gemma_export/gemma4_e2b`
on the conda `gemma4` env (torch 2.11.0 / transformers 5.7.0 / torchao
0.15.0+git9338966da):

| | wall | size | arena |
|---|---|---|---|
| xnnpack | 530s | 2.81 GB | 1.04 GB |
| vulkan | 174s | 2.57 GB | 144.6 MB |

`vulkan` lowered to **one delegate call** for `forward` — no fragmentation. Its
101 host ops are the PLE dequant/embedding lookups, which stay on CPU by design.

### What `method_meta` says vs the published config

Every constant-method matches its config exactly on both backends, including
the deliberate `get_max_seq_len` deviation (xnnpack bakes 4048 as the config
says, while the graph is 1024). The graph-method contract does NOT match, and
all three gaps were predicted above — they are now measured, not inferred:

| | xnnpack | vulkan |
|---|---|---|
| real `forward` inputs | `tokens` + `input_pos` | `tokens` + `input_pos` |
| config declares | `tokens` only | `tokens` only |
| real seq bound | 1024 | **128** |
| config claims | 1..2048 | 1..2048 |
| real logits dtype | float16 | float32 |
| config claims | float32 | float32 |

The seq over-claim is the sharp one: a runner honouring the published
`{"min":1,"max":2048}` will hand the Vulkan graph 16x more tokens than it
accepts. `multimodal_prefiller.cpp` bounds-checks nothing.

Reproduce with `program.metadata(name)` — it reads I/O without loading the
method, so `forward`'s arena is never allocated.
