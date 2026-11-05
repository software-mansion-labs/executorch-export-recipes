# lfm3_text

LiquidAI/LFM3-1.7B (instruction-tuned), pinned at revision
`40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`. 24 layers: 13 gated short convs
(kernel 7), 6 full-attention and 5 sliding-window (256) attention layers, 16
query / 4 KV heads of 128, vocabulary 128000 (the LFM2.5-2.6B tokenizer).

## Why the decoder is vendored

ExecuTorch's llama `Transformer` cannot express LFM3, so `_lfm3/model.py`
re-implements it for export (static KV caches and conv states as mutable
buffers, batch 1). What it lacks:

- a RoPE base per attention type (1e6 full, 1e4 sliding) and LFM3's zero-tail
  partial RoPE on full layers (the first 32 of 64 frequencies follow the
  full-head progression, the rest are 0, so `rotate_half` still pairs channel
  i with i + 64);
- the per-head sigmoid output gate (`gate_proj: hidden -> n_heads`);
- sandwich norm (`post_operator_norm`, `post_ffn_norm`) and the width-7 conv.

`attn_impl` picks the attention kernel per backend: `llama::update_cache` +
`llama::custom_sdpa` (XNNPACK) or `mlx::kv_cache_update` +
`mlx::custom_sdpa` (MLX). The sliding-window mask depends only on
(start_pos, T) and is built once per forward. On XNNPACK the sliding layers
read only the cache rows inside the window (`window_slice`).

The vendored model is exact against the upstream remote code in fp32: logit
cosine 1.00000 at every 128-token chunk boundary of a 646-token prompt (which
crosses the window twice) and 24/24 greedy tokens (`parity.py`).

## Contract

Context 8192, prefill chunk 128 (the `forward` token dim is a range 1..128 in
the baked `get_model_schema`, which RNE uses as its chunk size), float32
logits for the last token. Same values as LFM2.5-2.6B. KV cache at 8192: 11
attention layers x 2 x 16 MiB = 352 MiB fp32 (XNNPACK).

Each output dir also gets `tokenizer.json`, `tokenizer_config.json` with the
chat template embedded, and the upstream LICENSE. The template is patched to
strip thinking from every assistant turn (as for LFM2.5-2.6B and Qwen3):
upstream keeps it after the last user message, so appending a user turn
rewrites earlier output and breaks RNE's chat prefix check.

## Recipes and results

`validate.py` runs the .pte on the host ExecuTorch runtime against
`reference.py` (fp32 HF, greedy without the checkpoint's repetition penalty,
which `generate()` otherwise applies even with `do_sample=False`).

| Variant | Size | Delegates / host ops | cos @128 / @256 / @646 | teacher-forced |
| --- | ---: | --- | --- | ---: |
| XNNPACK 8da4w (gs32 linears, int8 per-row embedding) | 1.25 GB | 350 / RMSNorm, RoPE slices, sdpa | 0.834 / 0.716 / 0.986 | 21/24 |
| MLX int4 (gs64, embedding too) | 0.93 GB | 1 / none | 0.798 / 0.869 / 0.981 | 21/24 |
| MLX int8 (gs64) | 1.79 GB | 1 / none | 0.9994 / 0.9995 / 0.9999 | 24/24 |

The 8da4w loss is quantization, not lowering: the eager quantized model gives
the same cosines (0.856 / 0.729 / 0.986), and keeping the embedding or
`lm_head` unquantized changes nothing. It is the 4-bit body linears. The
low cosines sit at mid-prompt positions whose logits are flat; the answer
position is 0.98+. XNNPACK leaves RMSNorm and RoPE on portable kernels, as
the `export_llama` path does for every shipped LLM.

`to_executorch` needs `do_quant_fusion_and_const_prop=True`; without it the
int8 embedding leaves a `torchao::dequantize_affine` with no runtime kernel.

## No Vulkan export

Measured and dropped. With the fork's Vulkan SDPA taught to take `attn_mask`
(sliding layers pass one), pytorch/executorch#22479 (tensorless delegate
output) and the decode mask built from the `input_pos` tensor, the 1.7B lowers
to one delegate and generates correctly, but on a Galaxy S26 Ultra it decodes
at 31-39 tok/s against XNNPACK's 44-45, with equal short-prompt prefill (the
LFM2.5 text result again).

## Reproduce

```bash
# HF reference (separate env: the remote code needs transformers >= 5.20)
python exporter_models/lfm3_text/reference.py --source <snapshot> --output ref.pt --dtype fp32
# export (fork env); LFM3_SOURCE reuses a local snapshot instead of downloading
LFM3_SOURCE=<snapshot> PYTHONPATH=src python -m exporter.cli export lfm3_text \
  --size 1_7b --backend xnnpack --output-dir out/xnnpack
python exporter_models/lfm3_text/validate.py --model out/xnnpack/lfm_3_1_7b_xnnpack_8da4w.pte --reference ref.pt
```
