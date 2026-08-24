# phi_4_mini

## Why the config is vendored instead of fetched

The published `xnnpack/config.json` carries runner constants that do not belong
to this model:

| key | published | correct | evidence |
|---|---|---|---|
| `get_bos_id` | `151643` | `199999` | `151643` is Qwen's `<\|endoftext\|>` and is not a valid id in Phi's tokenizer's role; upstream `generation_config.json` declares `bos_token_id: 199999` |
| `get_eos_ids` | `[151643]` | `[200020, 199999]` | upstream declares `eos_token_id: [200020, 199999]` (`<\|end\|>`, `<\|endoftext\|>`) |
| `get_max_seq_len` | `128` | `2048` | contradicts the same config's `forward` input, which declares a sequence dim of `1..2048` |
| `get_max_context_len` | `128` | `2048` | same |

The bos/eos pair is Qwen2.5's, on a model whose vocab is 200064 -- the constants
were copied from a sibling repo. The old `exports/llms` recipe exported Phi-4-mini
with `{"get_bos_id":199999,"get_eos_ids":[199999]}` and
`--max-seq-length 2048 --max-context-length 2048`; the vendored config restores
those lengths and takes the full upstream eos pair. Everything else is
byte-identical to the published config.

Because `export.py` reads bos/eos and the sequence lengths from the config (the
single source of truth for runner constants), fetching would have exported a
Phi model carrying Qwen's stop tokens and a 128-token context.

## Re-publishing

The published `.pte` predates this fix, so a re-export is needed before the
corrected constants reach the runtime.
