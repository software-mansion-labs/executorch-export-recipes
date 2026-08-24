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

A dir whose children are models — not a model itself — groups related models
(e.g. a `family/` dir holding `family/detector` and `family/recognizer`, each
with its own `model.json` and contract). The registry discovers one level of
nesting: a top dir that isn't itself a model is scanned for child model dirs,
emitted as `parent/child`. The rest of the registry handles slash-bearing model
names via path joins. (No current model uses this: the old `easyocr/craft` +
`easyocr/crnn` pair was superseded by the fused, sized `easyocr` model.)

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
- export_llama path (lfm2_5_text, llama3_2, qwen2_5, qwen3, hammer2_1, smollm2,
  phi_4_mini): `cfg.base.metadata` is built from the config's
  `get_bos_id`/`get_eos_ids`, and `cfg.export.max_seq_length` /
  `max_context_length` from `get_max_seq_len` / `get_max_context_len`
  (export_llama derives vocab from params). These per-(size,backend) values
  genuinely differ (e.g. lfm2_5_text's `get_max_seq_len` is 2048 on xnnpack but
  65 on mlx), which is why they live in the config, not a shared constant.

When a model's config carries these, set them there — never re-hardcode in
export.py.

## export_llama models: params, weights, and the auto-download trap

The `export_llama` families keep a `_params.py` at the model-dir top holding, per
size, the architecture `params` (mirroring `examples/models/<m>/config/*.json`
where executorch ships one, carried locally where it does not), the
`model_class`, the upstream `source` checkpoint, and the `quant` recipe.

**Always set `cfg.base.checkpoint` explicitly.** Left empty, `export_llama`
resolves `model_class` through its own `HUGGING_FACE_REPO_IDS` table, which
points at the **base** weights (`Qwen/Qwen2.5-0.5B`,
`HuggingFaceTB/SmolLM2-135M`) while we publish the **Instruct** ones — and a
fine-tune that borrows another family's `model_class` (hammer2_1 rides on
qwen2_5) would silently export the wrong model entirely. `export.py` downloads
the pinned `source` and runs executorch's HF->Meta `convert_weights`, caching
the result under `~/.cache/exporter_meta_checkpoints/`. Llama is the exception:
Meta ships native-format `params.json` + `consolidated.00.pth`, so it needs no
conversion table at all. `lfm2_5_text` is the one model still relying on the
auto-download; it is safe only because the table happens to name the repos we
publish from.

## Per-model dependencies (pinned uv projects)

Each `exporter_models/<model>/` is its **own uv project** — a `pyproject.toml`
+ `uv.lock` that path-depends on the `exporter` harness and pins the model's
**exact, verified** stack (torch / executorch / transformers / torchao / …).
Run an export inside the model's locked env:

```bash
uv run --project exporter_models/<model> exporter export <model> --backend <b> --output-dir <dir>
```

Why per-model (not one shared env): the models genuinely conflict — the torch
2.14 nightly (matched to executorch 1.3.1) vs `rfdetr==1.8.2`'s `torch<=2.13`
pin, funasr's undeclared torchaudio, the MLX models' executorch source build.
A single lock can't satisfy them.

Conventions when authoring a model's `pyproject.toml`:
- `requires-python = "==3.10.*"`; repo-root `.python-version` is `3.10`.
- Torch-family (`torch`, `torchvision`, `torchaudio`, `torchao`) come from the
  PyTorch **nightly** index via `[tool.uv.sources] <pkg> = { index = "pytorch-nightly" }`
  + a `[[tool.uv.index]]` block. **Date-match** torch and torchvision (a nightly
  torchvision hard-pins its same-day torch). torchao is the pure-python
  `py3-none-any` nightly wheel (no C++ build).
- `[tool.uv] environments = ["sys_platform == 'darwin' and platform_machine == 'arm64'"]`
  pins the lock to this host and avoids the nightly +cpu/plain platform split.
- rfdetr/ultralytics under-constrain torch, so add
  `override-dependencies = ["torch==…", "torchvision==…"]` to force the nightly.
- `safetensors` is effectively always needed (the ET `export_llama` path reads
  it); funasr also needs `torchaudio`.
- MLX-backend models (gemma*/lfm2_5_vl) need executorch built from the
  software-mansion-labs fork (`branch = "@ms/separate-backends"`) — a native
  source build; their pyproject **declares** the fork source but is not
  uv-buildable here (run in the ET-source dev env; `.pte` ships on HF).

The `model.json` `extras` survive only as human-readable stack labels; the
`export` preflight (`registry.check_extras`) is a light presence sanity-check.
`executorch` and torch are pinned in each model's pyproject, not installed
separately.

## Weights are never committed

`.pt`/`.pth`/`.pte` are gitignored. Every model fetches/builds its weights at
export time (HF `from_pretrained` / `hf_hub_download`, URL-zip download,
keras->pytorch convert, ultralytics auto-download). Tests live in `tests/`
(also gitignored — local only).
