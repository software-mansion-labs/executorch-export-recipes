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

`forward` casts the logits to float32 (`_patch_fp32_logits`): bf16 logits fail in
react-native-executorch with "Unsupported ScalarType", and the other MLX LLMs
return float32.

Both published exports bake `get_model_schema` (`_params.model_schema`): without
it RNE reads a constant token dim from MethodMeta and rejects the model.

## Vulkan

int8 weight-only, per channel, via export_llama's `vulkan_8w` PT2E quantizer:
one delegate, 169 `_weight_int8pack_mm` + 24 fused `sdpa_with_kv_cache`, 1.76
GB. The torchao `8da8w` recipe is unusable here: Vulkan fuses only 4-bit torchao
weights, so each int8 linear keeps a `torchao.dequantize_affine` that the runtime
has no kernel for (the .pte would abort at load), and the activation quant is
dropped.

**Not published.** On a Galaxy S26 Ultra it decodes garbage from the first token, and is
slow even so: 93 ms/token, 79 of 83 ms of GPU time in `linear_q8csw_tiled`, which takes
as long at M=1 as at M=44 (no GEMV path for int8 weights). The garbage is fp16 overflow:
the PT2E path leaves RMSNorm decomposed (mul/mean/rsqrt), and EuroLLM's residual stream
has outliers up to |x| = 1749, so `x*x` overflows fp16 in 33 of 49 norms. The fork's
rms_norm fusion did not match this graph.

## Galaxy S26 Ultra (2026-10-22)

xnnpack 8da8w, 50 WMT13 en->de sentences, greedy, no BOS: BLEU 29.72, 28.5 ms/token
(35 tok/s), prefill ~565 tok/s, load 2.8 s. Opus-MT int8 does the same pair at
BLEU 29.13 in ~30 ms per sentence, so EuroLLM is for language coverage, not speed.

## Tokenizer streaming

The tokenizer's decoder ends with `Fuse` + `Strip`. pytorch/tokenizers decoded
a streamed token alone and stripped every leading space ("Helloworld"); fixed
in meta-pytorch/tokenizers#227. Score with collected ids, not the runner's
token callback, until that lands.

## Weights

`_params.py` reuses qwen3's HF->Meta converter: a plain Llama key map that keeps
the untied `lm_head` (SmolLM2's ties it to the embedding). `use_hf_rope` keeps
HF's rotate-half RoPE.

## iPhone 16 (2026-10-22)

50 WMT13 en->de sentences, 2 interleaved rounds, outputs identical across rounds:

| variant | prefill (mean, ~45 tok prompts) | decode | warm load | BLEU |
|---|---|---|---|---|
| mlx int8 | 137-237 ms | 33-36 ms/token | 0.11 s | 27.26 |
| xnnpack 8da8w | 356-372 ms | 52-53 ms/token | 1.1 s | 28.53 |

MLX is 1.55x faster to decode and ~2x faster to prefill for 1.3 BLEU; it is the iOS pick.
MLX ran fine through the Module API here, unlike RNE's LLMRunner on this device earlier.

