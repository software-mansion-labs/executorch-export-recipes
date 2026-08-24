# hammer2_1

## Why the config is vendored instead of fetched

The published repo ships `{size}/xnnpack/config.json` for all three sizes, but
its `get_eos_ids` is `[151643]` for every size. `151643` is `<|endoftext|>`;
the token a Qwen2.5-family chat model actually ends a turn on is `<|im_end|>`
(`151645`). A runner that only stops on `151643` runs past the end of the
assistant turn.

The upstream generation config
([`MadeAgents/Hammer2.1-0.5b`](https://huggingface.co/MadeAgents/Hammer2.1-0.5b/blob/main/generation_config.json))
declares `eos_token_id: [151645, 151643]`, and the old `exports/llms` recipe
exported these with `{"get_bos_id":151643,"get_eos_ids":[151645]}`. The vendored
config carries the upstream pair `[151645, 151643]`, which is a superset of both
and stops correctly either way. Everything else is byte-identical to the
published config.

The sibling `qwen2_5` model needs no such override -- its published config
already declares `[151645]` -- so it fetches.

## Architecture

Hammer2.1 is a function-calling fine-tune of Qwen2.5 with a smaller vocab
(151665 vs 151936) and is exported through the Qwen2.5 model classes and weight
converter. ExecuTorch ships no Hammer configs, so `_params.py` carries the
params for all three sizes.
