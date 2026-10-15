---
name: adding-a-new-export
description: How to add a model (or a backend/size to an existing one) to the exporter harness — create the model dir, declare model.json, source the config, write export.py implementing export(config, out_dir), and run it with `exporter export`.
---

# Adding a New Export

A model in the `exporter` harness is a self-contained directory under
`exporter_models/<model>/`. It owns its model code (vendored at the dir top)
and an `export.py` per backend. See `skills/model-conventions` for the full
`{size}`/`{backend}` layout, fetched-vs-local config rules, and nested models.

## Procedure

1. **Create the directory:** `exporter_models/<model>/<backend>/` (e.g.
   `.../xnnpack/`). Add a sibling `<backend>/` dir per backend you support.

2. **Write `model.json`** at the model root:

   ```json
   {
     "hf": "https://huggingface.co/software-mansion/react-native-executorch-<model>",
     "repo": "software-mansion/react-native-executorch-<model>",
     "config": "{backend}/config.json",
     "backends": ["xnnpack"]
   }
   ```

   - `hf` — the published HF repo URL (always set it).
   - `repo` — the repo the export fetches the config / weights from. Omit for
     a purely-local model. May differ from `hf` (e.g. an upstream source repo).
   - `config` — path template; use `{size}/{backend}/config.json` for a sized
     model and add `"sizes": [...]`. Put `{size}` in `repo` too if each size is
     a separate HF repo.

3. **Source the config.** If the HF `repo` already ships a correct
   `{backend}/config.json`, the registry fetches it — author nothing. Otherwise
   vendor a local `<model>/<backend>/config.json` (or `<size>/<backend>/...`).
   Required fields: `$schema`, `model`, `family`, `capabilities`, `backend`,
   `license`, `variants[]`. Each variant: `file` (output .pte name),
   `precision`, `quantized`, `default`, and per-method `{inputs,outputs}` with
   explicit `shape` + `dtype`. **Verify the shape contract against the real
   shipped `.pte`, not just the published config** — configs on HF are sometimes
   wrong (see `skills/backends/coreml`).

4. **Write `export.py`** in the backend subdir:

   ```python
   def export(config: dict, out_dir: str) -> None:
       """Build the model, export + lower, write the .pte(s) into out_dir.
       The return value is advisory; the harness reports results from the spec."""
   ```

   - Top-level imports: stdlib + torch only. Import executorch / transformers /
     torchao / coremltools / mlx and path-load vendored modules INSIDE `export()`
     — the harness imports `export.py` to discover it without a backend env.
   - Read from `config`: `config["variant"]["file"]` (output name),
     `config["variant"]["methods"][...]` (shapes), `config.get("size")` for
     sized models. Hardcode the source repo id in the vendored code as a
     constant (the convention), not via config.
   - Vendor model code at the model-dir top and path-load it (unique
     `sys.modules` key per dir/backend). Each model owns its code.
   - **Lower:** simple single-method XNNPACK models can use the shared
     `from exporter.backends import lower; lower(ep, config["backend"])`.
     Everything else lowers inline — see `skills/backends/{xnnpack,coreml,mlx}`.
   - Write the buffer to `Path(out_dir) / config["variant"]["file"]`.

5. **Author the model's uv project:** add `exporter_models/<model>/pyproject.toml`
   pinning the exact stack this model needs (see the "Per-model dependencies"
   section in `skills/model-conventions` for the full recipe — stable torch
   from PyPI (as facemesh and migan do), or a wheel URL if a nightly is
   unavoidable; matched torch/torchvision; `safetensors` always; environments
   pinned to this host; overrides for rfdetr/ultralytics). Then `uv lock` it and
   commit the `uv.lock`. The per-model `.venv` is gitignored. MLX and Vulkan
   backends are the exception: their pyproject declares the executorch fork pin
   but is not uv-buildable, and the export runs in a source-built dev env — see
   `skills/backends/{mlx,vulkan}`.

6. **Run it inside that env:**
   `uv run --project exporter_models/<model> exporter export <model> [--size S] [--backend B]`.
   The harness resolves the spec, runs your `export.py`, and writes the `.pte`.
   Use `exporter describe <model>` / `--dry-run` to inspect the plan first.
   Sanity of the produced `.pte` (I/O shapes, runtime parity) is the author's
   responsibility — check it against the authored `methods` / a reference run.
   Read "Export and verification traps" below before trusting a parity failure.

## Export and verification traps

Two mistakes that cost hours on migan and are invisible to the harness:

- **A numpy scalar stored on a module is traced as a tensor.** Upstream code
  that keeps `self.gain = np.sqrt(2)` (a `np.float64`) and later does
  `if act_gain != 1:` exports fine in eager but `torch.export` wraps the numpy
  scalar into a tensor, so the comparison becomes data-dependent control flow
  and the export fails on it. Patch the vendored module to keep a Python
  float (`self.gain = float(np.sqrt(2))`), mark the change (`# EXPORT PATCH`),
  and note it in the model's `NOTES.md`. Nothing to fix upstream; this is how
  `torch.export` treats numpy values. In-place ops (`leaky_relu(inplace=True)`,
  `add_`) are NOT a problem — export functionalises them, bit-identically.
- **A non-contiguous input makes every `.pte` look broken.** The ExecuTorch
  pybindings ignore tensor strides and read the raw buffer as contiguous NCHW
  (upstream bug: [pytorch/executorch#18562](https://github.com/pytorch/executorch/issues/18562),
  "`Method.execute()` silently produces wrong results for non-contiguous input
  tensors", open, assigned to a maintainer). The usual PIL -> numpy ->
  `torch.from_numpy(...).permute(2, 0, 1)` pipeline produces exactly such a
  tensor, so the runtime output is garbage while nothing errors — and the
  "parity failure" is the same for every variant you try, which is the tell.
  Confirm with the lowered program on the Python side
  (`program.exported_program().module()(*inputs)` matches eager while the
  loaded `.pte` does not), then call `.contiguous()` on every input before
  `execute`/`forward`. Until the runtime rejects strided inputs, treat
  `.contiguous()` as part of the verification recipe for every backend.

## Notes

- Each model pins ExecuTorch (and coremltools / MLX for those backends) in its
  own `pyproject.toml`; `uv run --project …` provisions the locked env.
- If a model is unusual (pinned transformers version, runtime op rebuild, a
  projection head, an iOS limitation), add a short `NOTES.md` beside the model
  documenting the quirk so re-export carries it.
- Tests live in `tests/` (gitignored, local) — mirror an existing
  `test_*_exports.py` (discovery, backends, sizes, IO shapes).
