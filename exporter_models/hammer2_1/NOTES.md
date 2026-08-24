# hammer2_1

## Architecture

Hammer2.1 is a function-calling fine-tune of Qwen2.5 with a smaller vocab
(151665 vs 151936), exported through the Qwen2.5 model classes and weight
converter. ExecuTorch ships no Hammer configs, so `_params.py` carries the params
for all three sizes.

## Config is vendored, not fetched

Beyond the two repo-wide errors (`get_n_layers: null`, sequence dim 2048 vs the
`.pte`'s 2047), the published config declares `get_eos_ids: [151643]` for every
size. `151643` is `<|endoftext|>`; the token a Qwen2.5-family chat model actually
ends a turn on is `<|im_end|>` (`151645`). A runner that stops only on `151643`
runs past the end of the assistant turn.

Upstream
([`MadeAgents/Hammer2.1-0.5b`](https://huggingface.co/MadeAgents/Hammer2.1-0.5b/blob/main/generation_config.json))
declares `eos_token_id: [151645, 151643]`, and the old `exports/llms` recipe
used `[151645]`. This config takes the upstream pair, which stops correctly
either way.

## Published artifacts

The shipped files disagree with each other -- the two variants of the same model
were exported with different stop tokens:

| variant | .pte `get_eos_ids` |
|---|---|
| `hammer_2_1_{0_5b,1_5b,3b}_xnnpack_bf16.pte` | `[151643]` |
| `hammer_2_1_{0_5b,1_5b,3b}_xnnpack_8da4w.pte` | `[151645]` |

So the published config matched only the bf16 half, and the bf16 half is the
broken one. All six need a re-export to carry `[151645, 151643]`; the 8da4w
files are usable in the meantime, the bf16 files are not.

Everything else (IO, bos, lengths, layer counts) matches. Check with
`tools/probe_published_pte.py hammer2_1 --size 0_5b`.

## Measured on device (Galaxy S26 Ultra, SM-S948B, Android 16)

The stop-token bug reproduces directly. Same prompt, greedy, executorch's
`llama_main` (arm64, ET 1.4.1 @226b471), 0_5b bf16:

```
published:  Three primary colors are red, blue, and yellow.<|im_end|>
            <|endoftext|>            <- ran past the turn, stopped only on 151643
re-exported: Three primary colors are red, blue, and yellow.<|im_end|>
                                     <- stopped on 151645, as intended
```

Identical text, but the published file emits `<|im_end|>` as content and keeps
generating. In a chat session that leaks the turn marker into the answer and
burns tokens until `<|endoftext|>` happens to appear.
