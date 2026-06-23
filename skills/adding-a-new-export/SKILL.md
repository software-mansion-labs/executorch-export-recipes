---
name: adding-a-new-export
description: How to add a new model to the exporter harness — create the model dir, author config.json, write export.py implementing export(config, out_dir), and verify with `exporter export`.
---

# Adding a New Export

A model in the `exporter` harness is a self-contained directory under
`exporter_models/<model>/` holding exactly two files: `config.json` (the spec)
and `export.py` (the engine). The harness runs the `export.py` sitting next to
the config — there is no resolver, no shared wrapper, no `family` field.

## Procedure

1. **Create the directory:** `exporter_models/<model>/`.

2. **Author `config.json`** to the schema. Required fields: `$schema`, `source`
   (upstream HF id), `model` (= the dir name), `capabilities` (e.g.
   `["text-embedding"]`), `backend` (`xnnpack` for now), `license`, and
   `variants[]`. Each variant: `file` (output .pte name), `precision`,
   `quantized`, `default`, and `methods.forward.{inputs,outputs}` with explicit
   `shape` + `dtype`. The input shape's last dim is the max sequence length.
   Copy an existing model's `config.json` (e.g. `all_minilm_l6_v2/`) as a
   starting point.

3. **Write `export.py`** implementing the contract:

   ```python
   def export(config: dict, out_dir: str) -> str:
       """Build the model, export+lower, write the .pte, return its path."""
   ```

   - Read what you need from `config`: `config["source"]`, `config["backend"]`,
     `config["variant"]["file"]` (output name),
     `config["variant"]["methods"]["forward"]["inputs"][0]["shape"][-1]` (max
     seq len).
   - Hardcode model-specific logic (pooling, CLS/SEP token ids, any custom
     wrapper) in this file — duplication across models is accepted; each script
     owns its model.
   - Lower via `from exporter.backends import lower; lower(exported_program, config["backend"])`.
   - Write the buffer to `Path(out_dir) / config["variant"]["file"]` and return
     that path as a string.
   - See `exporter_models/all_minilm_l6_v2/export.py` for a complete example, and
     `skills/backends/xnnpack/SKILL.md` for backend specifics (8da4w, tolerances).

4. **Verify:** run `exporter export <model> --json`. The harness invokes your
   `export.py`, then checks the produced .pte's real I/O matches the `methods`
   you authored (fails loudly on mismatch) and runs a runtime sanity + benchmark.
   Use `exporter describe <model> --json` / `exporter export <model> --dry-run`
   to inspect the plan without running.

## Notes

- ExecuTorch must be installed from source in the active venv to actually export.
- If a model is unusual (pinned transformers version, runtime op rebuild, a
  projection head, an ANE limitation), add a short `NOTES.md` beside its
  `config.json` documenting the quirk so re-export carries it.
