# phi_4_mini

## The published bf16 artifact is a different model's contract

Probing the shipped `.pte` explains why the published config looked so wrong:
it was written from the bf16 file, and that file was exported with Qwen's
metadata and a 128-token context.

| | `phi_4_mini_xnnpack_bf16.pte` | `phi_4_mini_xnnpack_8da4w.pte` | this config |
|---|---|---|---|
| `get_bos_id` | `151643` (Qwen `<\|endoftext\|>`) | `199999` | `199999` |
| `get_eos_ids` | `[151643]` | `[199999]` | `[200020, 199999]` |
| `get_max_seq_len` | `128` | `2048` | `2048` |
| `get_max_context_len` | `128` | `2048` | `2048` |
| `forward` seq dim | `127` | `2047` | `2047` |

Phi-4-mini's vocab is 200064, so `151643` is not even a meaningful id here.
Upstream
([`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/generation_config.json))
declares `bos_token_id: 199999` and `eos_token_id: [200020, 199999]`, where
`200020` is `<|end|>` -- the turn terminator. The old `exports/llms` recipe used
`{"get_bos_id":199999,"get_eos_ids":[199999]}` and exported at 2048/2048.

So the config here carries the upstream values, and:

- **bf16 needs a re-export.** As shipped it has Qwen's bos/eos and a 128-token
  context; it is not usable as a Phi-4-mini chat model.
- **8da4w needs a re-export too**, but only for `get_eos_ids`: it stops on
  `<|endoftext|>` and never on `<|end|>`, so generation runs past the end of the
  assistant turn. Everything else about it is correct.

Check with `tools/probe_published_pte.py phi_4_mini`.

## No size axis

Phi-4-mini ships in one size, so this model has no `{size}` placeholder and its
config lives at `xnnpack/config.json`.
