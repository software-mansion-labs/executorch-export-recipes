# export-scripts

A "script-as-engine" harness for exporting models to ExecuTorch `.pte` files.
Each model is a self-contained directory that owns its model code and a small
`export.py` per backend; the harness (`src/exporter/`) discovers models,
resolves their spec, runs the export, and verifies the produced `.pte` against
the authored I/O contract.

## Layout

```
exporter_models/<model>/
  model.json                 # pointer: hf url, repo, config path, sizes, backends
  <backend>/export.py        # the engine: export(config, out_dir) writes the .pte(s)
  <backend>/config.json      # local spec (only when not fetched from HF)
  <size>/<backend>/config.json   # for sized models
  _vendored_code.py          # model code, vendored at the model-dir top
```

- **`model.json`** is the per-model root metadata. Keys: `hf` (published HF
  repo URL), `repo` (source repo the export fetches from, when different),
  `config` (config path template with `{size}`/`{backend}` placeholders),
  `sizes` (optional), `backends`. A model can also be purely local (no `repo`).
- **Backend axis** — a model declares `backends: [...]` and puts an `export.py`
  in each `<backend>/` subdir. Adding a backend is additive.
- **Size axis** — a model declares `sizes: [...]` and a `{size}` placeholder in
  `config` (and/or `repo`). One `export.py` per backend reads `config["size"]`;
  there is no per-size `export.py`.
- **Config** is fetched from the HF `repo` when it ships a correct
  `{backend}/config.json`, otherwise vendored locally (no repo, or the HF config
  is wrong/missing/shaped differently). A local config under the resolved path
  wins over the fetch.
- **Vendored code** lives at the model-dir top (e.g. `_pf_model.py`,
  `_custom_whisper/`) and is path-loaded inside `export()`. Each model owns its
  code — duplication across models is accepted over cross-model imports.
- **Nested/umbrella models** — a dir whose children are models (e.g.
  `easyocr/craft`, `easyocr/crnn`) groups related models without being a model
  itself.
- **Dependencies** — each model pins its **own** exact, verified stack. Every
  `exporter_models/<model>/` is a standalone **uv project** (`pyproject.toml` +
  `uv.lock`) that path-depends on this harness and pins torch / executorch /
  transformers / torchao / … to the versions that produced its `.pte`. This is
  per-model because the models genuinely conflict (rfdetr forces torch<=2.13
  via an override; funasr needs torchaudio; the MLX models and the Vulkan
  exports need an executorch source build). Run an export inside a model's
  locked env:

  ```bash
  uv run --project exporter_models/<model> exporter export <model> --backend <b>
  ```

  The harness targets **python 3.10** (`.python-version`); from now on torch
  is a **stable release from PyPI** (as facemesh and migan do), or a **wheel
  URL** in `[tool.uv.sources]` when a nightly is unavoidable — no new dated pins
  on the nightly index, which gets pruned (most existing models still carry
  one, to be swapped at their next re-export) — and locks are pinned to this
  macOS arm64 host (`tool.uv.environments`). The old
  `model.json` `extras` remain as human-readable stack labels; `export` still
  preflights them as a light sanity check.

## CLI

Installed as `exporter` (see `pyproject.toml`):

```bash
exporter list [--json]
exporter describe <model> [--size S] [--backend B] [--json]
exporter export   <model> [--size S] [--backend B] [--variant P | --all-variants]
                          [--output-dir DIR] [--dry-run] [--json]
```

`export` resolves the spec, runs the model's `export.py`, and writes the
`.pte`(s). `--dry-run` prints the plan only.

## The export.py contract

```python
def export(config: dict, out_dir: str) -> None:
    """Build the model, export + lower, write the .pte(s) into out_dir.
    The return value is advisory; the harness reports results from the spec."""
```

Top-level imports must stay stdlib + torch only; executorch / transformers /
torchao / coremltools / mlx and vendored modules are imported inside `export()`
(the harness imports `export.py` to discover it without a full backend env).

## Development

The `src/exporter` harness is importable with `PYTHONPATH=src` (or
`pip install -e .`). Run the CLI with `exporter <command>` (see above). A test
suite lives under `tests/` but is local-only (gitignored, not in the repo);
if present, run it with `PYTHONPATH=src pytest -q tests/`.

ExecuTorch (and coremltools / MLX for those backends) must be installed in the
active environment to actually produce a `.pte`. Backend specifics and the
model conventions live in `skills/` (start with `skills/adding-a-new-export`).
