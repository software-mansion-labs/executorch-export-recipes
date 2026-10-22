# eurollm

EuroLLM-1.7B-Instruct (Apache-2.0, 35 languages), exported for translation
through export_llama. Prompt it the way the model card does:

```
<|im_start|>system\n<|im_end|>\n<|im_start|>user\nTranslate the following {Src} source text to {Tgt}:\n{Src}: {text} \n{Tgt}: <|im_end|>\n<|im_start|>assistant\n
```

## Quantization: 8da8w, not 8da4w

Greedy, WMT13 en->de, BLEU (host, XNNPACK):

| variant | size | 12 sent. no BOS / BOS | 50 sent. no BOS / BOS | ms/token |
| --- | --- | --- | --- | --- |
| fp32 eager (ET Llama module) | - | 31.50 / 30.40 | - | - |
| 8da8w per-channel, emb 8,0 | 1.66 GB | 26.58 / 29.06 | 28.18 / 26.84 | 16.9 |
| 8da4w gs32, emb 8,64 | 1.07 GB | 23.18 / 10.08 | 22.48 / 8.50 | 12.2 |

8da4w loses 6-9 BLEU and, with a BOS token, often fails to emit `<|im_end|>`
and keeps generating ("English: ... Das heißt: ..."). Translation quality is
the point of this model, so only 8da8w is published.

react-native-executorch renders the chat template without `bos_token` and its
tokenizer adds no specials, so the model runs without BOS there.

## MLX

export_llama's weight-only path is int4 only (`4w`), so `mlx/export.py` adds
an `8w` qmode: IntxWeightOnlyConfig int8, PerGroup(64), on linears and the
embedding. One delegate, every op supported. en->de, 50 sentences, no BOS:
BLEU 27.26, 13.8 ms/token on the host, 1.71 GB.

## Tokenizer streaming

The tokenizer's decoder ends with `Fuse` + `Strip`. pytorch/tokenizers decoded
a streamed token alone and stripped every leading space ("Helloworld"); fixed
in meta-pytorch/tokenizers#227. Score with collected ids, not the runner's
token callback, until that lands.

## Weights

`_params.py` reuses qwen3's HF->Meta converter: a plain Llama key map that keeps
the untied `lm_head` (SmolLM2's ties it to the embedding). `use_hf_rope` keeps
HF's rotate-half RoPE.
