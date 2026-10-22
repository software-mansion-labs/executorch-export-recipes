# opus_mt

Helsinki-NLP Opus-MT (MarianMT) translation pairs, one size per direction
(`en_de`, `de_en`, `en_es`, `es_en`, `en_fr`, `fr_en`). Every pair is the same
6+6-layer, d=512 architecture; only the vocabulary differs. Configs are generated
by `_gen_configs.py`; edit the table there, not the JSON.

## Contract

```
encode(input_ids i64[1,S], attention_mask i64[1,S]) -> 12 x f32[8,S,64]   (k_0, v_0, ..., k_5, v_5)
decode(input_ids i64[1,1], cache_position i64[1], attention_mask i64[1,S], 12 x f32[8,S,64]) -> f32[1,1,V]
```

`encode` returns the decoder's cross-attention K/V, so `decode` never recomputes
them (they were ~10x the rest of a decode step at a 64-token source). The
self-attention KV cache lives in the program (256 target positions). `S` is
dynamic 1..256 on XNNPACK/MLX/Vulkan and static 128 on Core ML (pad + mask).
The tokenizer adds no `</s>`: callers append it, then start decoding from the
pad id (Marian's decoder start token) and stop on `</s>`.

## Tokenizer

There is no upstream fast tokenizer, and the `Xenova/opus-mt-*` ones carry
`precompiled_charsmap: null`, which HF `tokenizers` refuses. `_tokenizer.py`
builds one from `source.spm` + `vocab.json`. On WMT13 (3000 sentences per pair)
it matches `MarianTokenizer` on 99.6-99.97% of inputs; the rest are Unigram
ties on accented subwords, or a source-only `<unk>` (`|`) becoming a target
piece. Decoded text matches.

## Accuracy and delegation (XNNPACK, host, greedy, WMT13 first 300)

| pair | fp32 BLEU | int8 BLEU | int8 identical to fp32 |
| --- | --- | --- | --- |
| en_de | 29.04 | 29.28 | 226/300 |
| de_en | 35.24 | 34.76 | 224/300 |
| en_es | 36.37 | 36.43 | 241/300 |
| es_en | 36.92 | 37.59 | 234/300 |
| en_fr | 49.04 | 49.02 | 233/300 |
| fr_en | 49.60 | 50.07 | 225/300 |

int8 (8da8w per-channel Linears incl. the output head, int8 embedding) halves
decode time at no accuracy cost: ~1.7-2.0 vs ~3.6 ms/step on an M-series host.
Greedy decoding throughout; HF's default for these models is 4 beams.

Delegation on XNNPACK: encode is 2 delegate calls, decode 7 (one per block,
split only by the in-place `index_put_` cache writes); 98-99% of execute time
is inside the delegate. Three things were needed to get there:

- `run_reinplace_pass`: without it each cache write was a functional
  `index_put` plus a `copy_` back, 16% of a decode step.
- A decomposed LayerNorm (`_LayerNorm.decomposed`): XNNPACK has no layer_norm
  and only reduces the two innermost dims of a 4D tensor.
- Reshapes spelled with `-1` and per-block K/V tensors: XNNPACK rejects
  symbolic reshapes and slices of dynamic-shaped inputs.

Encode outputs are `.contiguous()`: a transposed view is recorded as dim order
(1, 0, 2), and feeding such an input to XNNPACK fails
(pytorch/executorch#24081).

## Core ML

fp16 decode is wrong (BLEU 4): Core ML's fp16 state for the KV cache breaks
from the second step on every compute unit. A plain coremltools repro shows the
same, so this is not ExecuTorch. fp32 is exact. On the host, fp32 decode costs
5.4 ms/step on CPU and the encoder 10-12 ms, both slower than XNNPACK int8.

So the one shipped variant is fp16 `encode` (ALL) + fp32 `decode` (CPU_ONLY).
en_de, 100 WMT13 sentences, host:

| variant | size | BLEU | identical to XNNPACK fp32 | encode | decode/step |
|---|---|---|---|---|---|
| fp16 enc + fp32 dec | 313 MB | 29.40 | 98/100 | 2.1 ms | 3.5 ms |
| fp32 | 298 MB | 29.23 | 100/100 | 8.2 ms | 3.8 ms |

It is larger than fp32 because the two functions no longer share the embedding.

## MLX

One delegate per method (`mlx::kv_cache_update` + `mlx::custom_sdpa` for the
cache). bf16 and int8 are accurate (BLEU 30.3 / 30.6 on en_de, 30 sentences),
but a single-token step is dispatch-bound on the GPU: 3.6-4.0 ms/step on the
host, against ~1.8 for XNNPACK int8.

## Vulkan

Encode is one Vulkan fp16 delegate; decode stays on XNNPACK int8 (one token,
latency bound, and the 58k-wide logits row exceeds a texture). Not verifiable
on the host; needs an Android device run before it can be the Android default.

## All pairs: Core ML and MLX quality

WMT13, first 100 sentences, host. In parentheses: outputs identical to XNNPACK
fp32 out of 100. Every backend is within ~1 BLEU of fp32 on every pair.

| pair | xnnpack fp32 | xnnpack int8 | coreml fp16 | mlx bf16 | mlx int8 |
|---|---|---|---|---|---|
| en_de | 29.23 | 29.32 | 29.40 (98) | 29.26 (94) | 29.13 (77) |
| de_en | 38.71 | 38.19 | 38.73 (98) | 37.67 (83) | 37.64 (81) |
| en_es | 35.94 | 36.20 | 35.88 (97) | 35.86 (93) | 36.05 (85) |
| es_en | 39.89 | 40.43 | 40.36 (96) | 40.18 (87) | 40.28 (82) |
| en_fr | 56.96 | 56.89 | 56.96 (99) | 57.32 (94) | 57.14 (90) |
| fr_en | 52.75 | 53.93 | 53.01 (97) | 53.41 (91) | 53.83 (84) |

Host medians across pairs: Core ML encode 2.0-2.2 ms, decode 3.1-4.3 ms/step;
MLX encode 3.5-5.4 ms, decode 2.5-3.6 ms/step. Plans are the same for every
pair: Core ML and MLX one delegate per method; Vulkan encode one Vulkan
delegate (+ host embedding lookup), decode 7 XNNPACK delegates.

## Galaxy S26 Ultra (2026-10-22)

`xbench` (Module API greedy encode/decode, fork 1.4.1 runtime, md5 d21dcf92), 100 WMT13
sentences, 2 warmup. en_de, medians of 4 interleaved rounds (order reversed each round):

| variant | encode | decode/step | load | BLEU |
|---|---|---|---|---|
| xnnpack int8 | 1.66 ms | 1.39 ms | ~105 ms | 29.13 |
| xnnpack fp32 | 3.6 ms | 3.9 ms | ~115 ms | 29.23 |
| vulkan fp16 | 6.0 ms | 1.42 ms | ~280 ms | 29.26 |

Outputs were identical across rounds for every arm. **Vulkan is not published**: its
decoder is the same XNNPACK int8 graph, and the GPU encoder is 3.6x slower than XNNPACK
int8 on these short (~20 token) sources, so it only adds load time.

xnnpack int8, other pairs (BLEU / encode / decode per step): de_en 38.48 / 2.43 / 1.45,
en_es 36.29 / 2.28 / 1.48, es_en 40.17 / 2.16 / 1.49, en_fr 56.86 / 1.71 / 1.41,
fr_en 53.46 / 1.88 / 1.44 ms.

