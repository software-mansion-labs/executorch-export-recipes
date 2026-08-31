# qwen3

## Config is vendored, not fetched

The published `{size}/xnnpack/config.json` got two things wrong for every size,
so a corrected copy lives here:

- `get_n_layers: null` where the `.pte` carries the real count (28 / 28 / 36).
- the `forward` sequence dim declared `max: 2048` where the `.pte` declares
  2047 (`max_seq_len - 1`).

Both are repo-wide across the LLM families, not specific to Qwen3. Everything
else matched.

## Published artifacts

All six `.pte` match this config. Verified with:

```bash
uv run --project exporter_models/qwen3 python tools/probe_published_pte.py qwen3 --size 0_6b
```

Nothing to re-export.
