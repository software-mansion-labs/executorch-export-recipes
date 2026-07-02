# CLAUDE.md

Guidance for Claude Code (and any agent) working in this repo. Read this first,
then the `skills/` for the task at hand.

## What this repo is

A "script-as-engine" harness for exporting models to ExecuTorch `.pte` files.
Each model is a self-contained directory under `exporter_models/<model>/` that
owns its model code and a small `export.py` per backend; the harness
(`src/exporter/`) discovers models, resolves their spec, runs the export, and
verifies the produced `.pte` against the authored I/O contract. See `README.md`
for the full narrative.

## Layout

```
exporter_models/<model>/
  model.json                    # pointer: hf url, repo, config path, sizes, backends, extras
  <backend>/export.py           # the engine: export(config, out_dir) -> pte path
  <backend>/config.json         # local spec (only when not fetched from HF)
  <size>/<backend>/config.json  # for sized models
  _vendored_code.py             # model code, vendored at the model-dir top
src/exporter/                   # the harness (registry, spec, pipeline, backends, cli)
skills/                         # how-to knowledge (start here for any task)
```

## CLI

Installed as `exporter` (see `pyproject.toml`); or run with `PYTHONPATH=src`.

```bash
exporter list [--json]
exporter describe <model> [--size S] [--backend B] [--json]
exporter export   <model> [--size S] [--backend B] [--variant P | --all-variants]
                          [--output-dir DIR] [--dry-run] [--json]
```

## The export.py contract

```python
def export(config: dict, out_dir: str) -> str:
    """Build the model, export + lower, write the .pte, return its path."""
```

Top-level imports stay stdlib + torch only; executorch / transformers / torchao
/ coremltools / mlx and vendored modules are imported INSIDE `export()` (the
harness imports `export.py` to discover it without a full backend env).

## Where to look (skills/)

- **`skills/adding-a-new-export`** — the step-by-step for adding a model, or a
  backend/size to an existing one. Start here.
- **`skills/model-conventions`** — `model.json`, the `{size}`/`{backend}` axes,
  fetched-vs-local config, vendored code, nested models, extras, LLM constants.
- **`skills/backends/{xnnpack,coreml,mlx}`** — per-backend lowering recipes,
  quantization, gotchas, and how to **profile / verify delegation** of a
  produced `.pte` with the executorch-profiler.

## Verifying an export

An export is not done when it produces a `.pte` — it is done when the `.pte`
matches its authored contract and lowers as intended:

1. **I/O shapes** — load the real `.pte` and check `method_meta` against the
   config's `methods` (published HF configs are sometimes wrong — see
   `skills/backends/coreml`). Never trust the config alone.
2. **Delegation & timing** — run the export-scripts sibling profiler,
   [executorch-profiler](https://gl.swmansion.com/swm-ai/executorch-profiler),
   to confirm ops actually land on the intended delegate (XNNPACK / CoreML)
   instead of falling back to CPU portable kernels, and to see per-op time. Each
   backend skill has a "Profiling & delegation check" section wiring this up.

## Conventions

- Weights (`.pt`/`.pth`/`.pte`) are gitignored — every model fetches/builds its
  weights at export time. Never commit them.
- Tests live under `tests/` but are local-only (gitignored, not in the repo).
- Per-model export deps are pip **extras** in `pyproject.toml`; a `model.json`
  names the extras it needs. `executorch` (and MLX for the mlx backend) are
  environment prerequisites installed separately, not extras.
- Each model owns its code — duplication across models is accepted over
  cross-model imports.
