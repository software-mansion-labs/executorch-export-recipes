# qwen2_5

## Config is vendored, not fetched

The published `{size}/xnnpack/config.json` got the two repo-wide things wrong:
`get_n_layers: null` where the `.pte` carries 24 / 28 / 36, and a `forward`
sequence dim declared `max: 2048` where the `.pte` declares 2047
(`max_seq_len - 1`). Corrected here; the rest matched.

## Published artifacts

Five of the six shipped `.pte` match this config. The exception:

| file | config | .pte |
|---|---|---|
| `qwen_2_5_3b_xnnpack_8da4w.pte` | `get_eos_ids [151645]` | `[151643]` |

`151645` is `<|im_end|>`, the token a Qwen2.5 chat model ends a turn on;
`151643` is `<|endoftext|>`. The 3B bf16 artifact and both smaller sizes all
carry `[151645]`, and so does the config -- only this one file differs. It comes
straight from the old `exports/llms` README, whose 3B recipe alone passed
`"get_eos_ids":[151643]` while 0.5B and 1.5B passed `[151645]`; that looks like a
copy error in the recipe rather than a decision. The config keeps `[151645]`, so
`qwen_2_5_3b_xnnpack_8da4w.pte` needs a re-export.

Check with `tools/probe_published_pte.py qwen2_5 --size 3b`.
