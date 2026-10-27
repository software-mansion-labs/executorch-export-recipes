# MiniCPM5-2B

Exports the final post-trained `openbmb/MiniCPM5-2B` checkpoint at
`f97400052a43d642bbc6e9975e2397e3ae6a6b52`, using ExecuTorch's standard Llama
implementation with Hugging Face RoPE. Both backends expose float32 logits,
position-addressed KV caches, an 8,192-token context, and dynamic prefill of
1–128 tokens. Linear weights and embeddings are quantized to int4 (group 64);
XNNPACK also quantizes activations dynamically to int8.

The tokenizer config selects the upstream thinking prefix (`<think>\n`).
Reasoning is stripped from every historical assistant message, independently of
later turns, to keep history monotonic.
The upstream XML tool serialization is preserved. Its two tokenizer regex splits
are fused into one equivalent split for the C++ tokenizer, retaining three-digit
grouping and whitespace boundaries before digits.

Run exports **sequentially**:

```sh
uv run --project exporter_models/minicpm5 exporter export minicpm5 --backend xnnpack --output-dir /tmp/minicpm5/xnnpack
MINICPM5_EXECUTORCH_ROOT=/path/to/executorch uv run --project exporter_models/minicpm5 exporter export minicpm5 --backend mlx --output-dir /tmp/minicpm5/mlx
```

MLX requires the Software Mansion ExecuTorch fork's MLX Python backend and its
native delegate. `MINICPM5_EXECUTORCH_ROOT` selects a source checkout for the
Python backend; the checkpoint and export environment are otherwise pinned.

## Validation

Tested sequentially on an M1 Pro Mac with native ExecuTorch runtimes. The
float32 conversion matches Hugging Face within 0.00025 per logit; MLX matches
the quantized PyTorch reference within 0.00009. XNNPACK's dynamic activation
quantization introduces further rounding differences; sampled argmaxes and
the greedy answer match the original checkpoint. Both backends pass arithmetic,
Chinese, exact multilingual copying, two-turn recall, and cache rollback.
Thinking-enabled two-turn sessions and monotonic rendered prefixes also pass
on both backends.

Full 8,192-token prefill (chunks of 128) and final-chunk replay pass on both:
XNNPACK 241.5 seconds, MLX 21.3 seconds. These are local smoke measurements,
not mobile benchmarks. Profiling confirms XNNPACK linear delegation and one
MLX decoder partition; the int4 embedding lookup remains a CPU operation.
