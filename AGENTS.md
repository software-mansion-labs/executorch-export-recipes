# AGENTS.md

Guidance for coding agents working in this repo. Read this first,
then the `skills/` for the task at hand.

## What this repo is

A "script-as-engine" harness for exporting models to ExecuTorch `.pte` files.
Each model is a self-contained directory under `exporter_models/<model>/` that
owns its model code and a small `export.py` per backend; the harness
(`src/exporter/`) discovers models, resolves their spec, and runs the export.
Runtime contract checks, numerical parity, and device benchmarks are separate
validation steps. See `README.md`
for the full narrative.

## Layout

```
exporter_models/<model>/
  model.json                    # pointer: hf url, repo, config path, sizes, backends, extras
  <backend>/export.py           # the engine: export(config, out_dir) writes the .pte(s)
  <backend>/config.json         # local spec (only when not fetched from HF)
  <size>/<backend>/config.json  # for sized models
  _vendored_code.py             # model code, vendored at the model-dir top
src/exporter/                   # the harness (registry, spec, pipeline, backends, cli)
skills/                         # how-to knowledge (start here for any task)
tools/executorch-profiler/       # XNNPACK timing and CoreML compute-device plans
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
def export(config: dict, out_dir: str) -> None:
    """Export and lower the model; write the configured .pte file(s)."""
```

The return value is advisory; the harness reports the spec's output names.

Top-level imports stay stdlib + torch only; executorch / transformers / torchao
/ coremltools / mlx and vendored modules are imported INSIDE `export()` (the
harness imports `export.py` to discover it without a full backend env).

## Where to look (skills/)

- **`skills/adding-a-new-export`** — the step-by-step for adding a model, or a
  backend/size to an existing one. Start here.
- **`skills/model-conventions`** — `model.json`, the `{size}`/`{backend}` axes,
  fetched-vs-local config, vendored code, nested models, extras, LLM constants.
- **[Export recovery](skills/recover-export/SKILL.md)** — localize failures,
  make evidence-based retries, and validate the recovered artifact.
- **[Validation and optimization](skills/validate-and-optimize/SKILL.md)** —
  preserve output quality while measuring and improving target-device speed.
- **`skills/backends/{xnnpack,coreml,mlx,vulkan}`** — per-backend lowering
  recipes, quantization, gotchas, and how to **profile / verify delegation** of
  a produced `.pte` with a compatible event-traced runtime and developer tools.

Also see `docs/executorch-1.4.1-bump.md` for the ExecuTorch 1.4.1 pin bump: which
models are byte-identical across ET versions (so a re-export is a no-op), the
`torchao` `.dev`-sorts-below-release trap, and why CoreML `.pte` cannot be verified
in-process.

## Verifying an export

Follow [validate-and-optimize](skills/validate-and-optimize/SKILL.md). Check the
actual artifact's method metadata and constants, then compare outputs with the
original eager model using fixed representative inputs and declared tolerances.
For quantized models, check task quality against the original source as well.
Profile delegation and benchmark on the target runtime/device; report which
checks passed and which could not run. The CLI does not perform these checks.
For the bundled profiler's setup and usage, see the
[profiler guide](skills/validate-and-optimize/references/profiling.md).
If export or runtime execution fails, use
[recover-export](skills/recover-export/SKILL.md) before retrying.

## Conventions

- Do not add generated weights or `.pte` files. Outputs are gitignored;
  historical large files use LFS and a few existing models have explicit
  vendored-data exceptions. Preserve their provenance.
- Original project code is MIT. Retain vendored copyright/license notices and
  model weight terms; see [docs/third-party-notices.md](docs/third-party-notices.md).
- Tests live under `tests/` but are local-only (gitignored, not in the repo).
- Per-model export deps are **pinned per model**: each `exporter_models/<model>/`
  is its own uv project (`pyproject.toml` + `uv.lock`) that path-depends on the
  harness and pins its exact stack (torch/executorch/transformers/…). Run inside
  it with `uv run --project exporter_models/<model> exporter export <model> …`.
  From now on torch is a stable release from PyPI (as facemesh does), or a
  wheel URL when a nightly is unavoidable; do not add new dated pins on the
  nightly index, which gets pruned and stops resolving (most existing models
  still have one, swap it at their next re-export). Python is pinned to 3.10;
  locks target this macOS arm64 host. The `model.json` `extras` survive as
  stack labels only. rfdetr/ultralytics need a torch `override-dependencies`.
  The MLX models gemma*/lfm2_5_vl/sdxs and every Vulkan export need an
  executorch source build from the software-mansion-labs fork (its CMake
  requires a checkout dir named `executorch`, which uv cannot provide) and are
  documented, not uv-built.
- Each model owns its code — duplication across models is accepted over
  cross-model imports.

## README and artwork

Edit `README.md` directly. Its CLI, layout, and export contract are part of the
public documentation. The banner is a standalone SVG at
`.github/assets/export-recipes.svg`; no JSX, generator, or Node tooling is
required. Preserve its accessible title/description and official React Native
ExecuTorch flame. Keep the artwork primarily black
and white with blue accents and gallery-style Helvetica Neue/system sans-serif
typography.
