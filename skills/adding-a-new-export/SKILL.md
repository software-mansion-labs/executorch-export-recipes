---
name: adding-a-new-export
description: How to add a model (or a backend/size to an existing one) to the exporter harness — create the model dir, declare model.json, source the config, write export.py implementing export(config, out_dir), and verify with `exporter export`.
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
   def export(config: dict, out_dir: str) -> str:
       """Build the model, export + lower, write the .pte, return its path."""
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
   - Write the buffer to `Path(out_dir) / config["variant"]["file"]`, return it.

5. **Verify:** `exporter export <model> [--size S] [--backend B] [--json]`. The
   harness runs your `export.py`, then checks the produced `.pte`'s real I/O
   matches the authored `methods` (fails loudly) and runs a runtime sanity +
   benchmark. Use `exporter describe <model>` / `--dry-run` to inspect the plan.

## Notes

- ExecuTorch (and coremltools / MLX for those backends) must be installed to
  actually export.
- If a model is unusual (pinned transformers version, runtime op rebuild, a
  projection head, an iOS limitation), add a short `NOTES.md` beside the model
  documenting the quirk so re-export carries it.
- Tests live in `tests/` (gitignored, local) — mirror an existing
  `test_*_exports.py` (discovery, backends, sizes, IO shapes).
