---
name: model-conventions
description: Exporter harness model layout conventions — model.json (hf/repo/config/sizes/backends), the {size} and {backend} axes, fetched-vs-local config rules, vendored code, and nested/umbrella models.
---

# Model Conventions

How models are laid out and resolved in `exporter_models/`. For the step-by-step
of adding one, see `skills/adding-a-new-export`; for backend lowering, see
`skills/backends/*`.

## model.json (the per-model root)

```json
{
  "hf": "https://huggingface.co/software-mansion/react-native-executorch-<model>",
  "repo": "software-mansion/react-native-executorch-<model>",
  "config": "{size}/{backend}/config.json",
  "sizes": ["350m", "1_2b"],
  "backends": ["xnnpack", "mlx"]
}
```

- **`hf`** — the published HF repo URL. Set on every model.
- **`repo`** — the repo the export fetches config/weights from. May differ from
  `hf` (e.g. privacy filters fetch the upstream `openai/privacy-filter` while
  `hf` is the SWM publish repo). Omit entirely for a purely-local model.
- **`config`** — path template resolved with the chosen size/backend.
- **`sizes` / `backends`** — the two axes (below).

## The {backend} axis

A model declares `backends: [...]` and puts `export.py` in each `<backend>/`
subdir (`<model>/xnnpack/export.py`, `<model>/coreml/export.py`, ...). Adding a
backend is purely additive — no existing file changes. The registry resolves
`export_script_path(model, backend)` to `<model>/<backend>/export.py`.

## The {size} axis

A model declares `sizes: [...]` and a `{size}` placeholder in `config`. There is
**one `export.py` per backend** — it reads `config["size"]` and branches
(params, model id, checkpoint). Do NOT make a per-size `export.py`. Two shapes:

- size in the **config path** only (`{size}/{backend}/config.json`), one HF repo
  holding all sizes — e.g. fast_sam.
- size in the **repo** too (`...-whisper-{size}`), one HF repo per size — the
  registry substitutes `{size}` in `repo` as well as `config`.

## Fetched vs local config

For each (size, backend) the registry first looks for a LOCAL config at the
resolved pointer path `<model>/<config_path>`; if absent, it FETCHES
`config.json` from the HF `repo`. Vendor a local config only when:

1. there is no `repo` (purely-local model), or
2. the HF repo has no config for that backend (e.g. kokoro, lfm2_5_vl,
   moonshine, rfdetr_keypoint), or
3. the HF config is wrong or shaped differently from what we export — e.g.
   ssdlite320 coreml (HF shape was wrong; verified against the real .pte), or a
   fully-dynamic `-1` dim we represent as a concrete `Dim(1,N)` (paraphrase).

Otherwise let it fetch — most models do. Don't vendor a redundant copy of a
config the repo already ships correctly.

## Vendored code

Model code lives at the model-dir top (`_pf_model.py`, `_custom_whisper/`,
`_lfm2_5_vl/`, ...) and is path-loaded inside `export()` with a unique
`sys.modules` key per dir/backend. Each model owns its code — copy shared code
into each model rather than importing across model dirs. Hardcode the source
HF id as a constant in the vendored code (like whisper's `openai/whisper-<size>`
or lfm's model_class), not via config.

## Nested / umbrella models

A dir whose children are models — not a model itself — groups related models.
`easyocr/` holds `easyocr/craft` and `easyocr/crnn`, each with its own
`model.json` and contract (craft is `multi_pte`, crnn is sized). The registry
discovers one level of nesting: a top dir that isn't itself a model is scanned
for child model dirs, emitted as `parent/child`. The rest of the registry
handles slash-bearing model names via path joins.

## LLM runner constants

The published spec config declares LLM runner metadata as non-IO method
entries with literal values — `get_bos_id: 2`, `get_eos_ids: [1,106,50]`,
`get_max_seq_len`, `get_max_context_len`, `get_vocab_size`, `use_kv_cache`,
`enable_dynamic_shape`, `has_ple`, `image_placeholder_id`, ... (int / list /
bool / null). The spec loader collects every method entry that lacks
`inputs`/`outputs` into `Variant.constants`, and the pipeline passes it through
as `config["variant"]["constants"]`. LLM exports embed these into the `.pte`
from the config (the single source of truth) — NOT hardcoded:

- MLX (gemma4, gemma4_omni): `constant_methods=config["variant"]["constants"]`.
- export_llama path (lfm2_5_text): `cfg.base.metadata` is built from the
  config's `get_bos_id`/`get_eos_ids` (export_llama derives seq_len/vocab from
  params). These per-(size,backend) values genuinely differ (e.g. text
  `get_max_seq_len` is 2048 on xnnpack but 65 on mlx), which is why they live
  in the config, not a shared constant.

When a model's config carries these, set them there — never re-hardcode in
export.py.

## Per-model dependencies (extras)

Base deps (torch, torchvision, transformers, huggingface_hub,
sentence-transformers) are always installed. Anything else a model needs to
export — coremltools, torchao, rfdetr, ultralytics, diffusers, funasr, keras,
... — is grouped into a pip **extra** in `pyproject.toml`
(`[project.optional-dependencies]`: `coreml`, `quant`, `detection`,
`diffusion`, `asr`, `vad`, `tts`, `moonshine`). A model declares the extras it
needs in `model.json`:

```json
{ "hf": "...", "extras": ["coreml", "detection"], "config": "...", "backends": [...] }
```

Install a model's stack with `pip install -e ".[coreml,detection]"`. The
`export` command runs a **preflight** (`registry.check_extras`) before invoking
`export.py`: a missing package fails early with the exact `pip install` hint
instead of a deep import crash. `--dry-run` skips the check. When you add a new
extra, update both `pyproject.toml` and the `_EXTRA_IMPORTS` map in
`registry.py` (pip name != import name for some, e.g. `keras-hub` -> `keras_hub`).
`executorch` (and MLX for the mlx backend) are environment prerequisites
installed separately, not pip extras.

## Weights are never committed

`.pt`/`.pth`/`.pte` are gitignored. Every model fetches/builds its weights at
export time (HF `from_pretrained` / `hf_hub_download`, URL-zip download,
keras->pytorch convert, ultralytics auto-download). Tests live in `tests/`
(also gitignored — local only).
