# Qwen3.5 text exports

These exports extract the text decoder from the post-trained Qwen3.5
checkpoints, not the Base models. Source revisions and architecture parameters
are pinned in `_params.json`. Vision weights are excluded by ExecuTorch's
Qwen3.5 converter.

The upstream dense Qwen3.5 implementation currently exports static,
single-token inputs. Prefill therefore runs one token at a time. The context
limit is 2048 tokens; this is an export limit, not the model's trained limit.
XNNPACK uses 8da4w linears, MLX uses int4 linears, and both quantize embeddings
with group size 64. Recurrent arithmetic remains FP32.

The tokenizer configuration embeds the upstream chat template with thinking
removed from every historical assistant message, preserving the prefix when
another user message is appended. System-only initialization is also allowed.
The template's native generation prompt disables thinking by default.

## Runtime compatibility

Gated DeltaNet's exported graph tracks the next token position. A rewind
restores each layer's convolution and recurrent state from its cached hidden
inputs before consuming the replacement token. Position zero clears state.
The standard `forward(tokens, input_pos)` interface is unchanged, so the library
runner needs no model-specific metadata or replay logic.

Caching hidden inputs costs 144 MiB for 0.8B and 288 MiB for 2B at 2,048 tokens.
Sequential calls
skip replay; rewinds replay only recurrent state transitions, without repeating
feed-forward layers, full attention, or the vocabulary projection. Replay cost
grows with the retained prefix. Export uses conditional/scan control flow and
unwraps quantized tensor subclasses before tracing those branches. XNNPACK
compiles each complete recurrent step once and reuses it for inference and
replay, so restoration uses identical state-update kernels. MLX shares its QKV,
decay, and update-gate projections between the two paths.
Parameters are frozen for inference before lowering; otherwise AOT attempts
to build backward graphs through the recurrent scan.
During export, debug provenance is limited to one generation to avoid a
PyTorch 2.14 recursive-copy stack overflow across nested control flow. This
scoped workaround is restored when export finishes and changes no tensor math.
The optional LLM quant-fusion/constant-propagation pass is disabled: its subgraph
rewriter corrupts nested scan modules in this version. Quantized linear kernels
are already compiled into their delegates before that pass.
XNNPACK's conditional tensor outputs are emitted as copies into their planned
buffers. ExecuTorch's default reference moves retain aliases to branch scratch
storage that subsequent layers overwrite across forward calls. This workaround
is scoped to export and requires no runtime changes.

MLX export needs the Software Mansion ExecuTorch MLX backend from the
`@ms/separate-backends-1.4.1` source tree, as with the other MLX LLM exports.
Run with that backend available on `executorch.backends.__path__`, and use
a native runtime built from the same source to validate on this Mac.

## Reproducing the exports

The locked uv project uses PyTorch 2.14.0 and ExecuTorch 1.4.1. The MLX export
was tested with the Software Mansion fork at
`814c42d7dbe2309c322ff1d9ff9017111131ce90`, plus `executorch-mlx.patch`.
The patch preserves Qwen's query gate and folds unit-offset RMS normalization
in FP32 before replacing it with native RMS normalization. It also keeps mutable
buffers shared across MLX partitions under runtime ownership, so conditional
state restoration cannot detach an update from its buffer.

```sh
uv run --project exporter_models/qwen3_5 exporter export qwen3_5 \
  --size 0_8b --backend xnnpack --output-dir /tmp/qwen3_5/xnnpack

# Apply executorch-mlx.patch to the fork checkout first.
QWEN3_5_EXECUTORCH_ROOT=/path/to/executorch \
  uv run --project exporter_models/qwen3_5 exporter export qwen3_5 \
  --size 0_8b --backend mlx --output-dir /tmp/qwen3_5/mlx
```

## Validation on this Mac

Both **0.8B and 2B** have been exported, executed, and registered in the library
and example app for XNNPACK and MLX. The 4B configurations are prepared but
have not been exported, tested, registered, or published by this change.

Logits were checked at prompt positions 0, 1, 5, 18, and 36:

- The unquantized PyTorch decoder matches the original Hugging Face FP32 model
  within `3e-5` maximum absolute logit error for 0.8B and `5.9e-5` for 2B.
- A separate FP32 XNNPACK export matches Hugging Face within `6.2e-5`.
  This 3.8 GiB diagnostic artifact is not published.
- MLX int4 matches the identically quantized PyTorch decoder within `7.1e-5`
  for 0.8B and `1.3e-4` for 2B.
- XNNPACK 8da4w has additional dynamic-activation quantization differences:
  minimum cosine similarity to its quantized PyTorch reference is `0.982` for
  0.8B and `0.978` for 2B; maximum absolute error reaches `3.69`. The selected
  top token matches at all five checked positions for both sizes. Quantized
  logits also differ from the original FP32 model; this is not FP32 parity.

Both backends are validated with native forward calls on this Mac, including
zero resets, nonzero rewinds, jumps within a retained prefix, replaced history,
and rollback to the original suffix. These checks compare against a fresh
sequential prefill and require no caller-side token replay.
XNNPACK reproduces those logits exactly; MLX's largest error across the state
restoration checks is `2.9e-5`.

The singleton sequence schema, Jinja system-only rendering, and monotonic
assistant history are checked separately. Greedy smoke prompts cover copying,
arithmetic, JSON, translation, and multi-turn recall; these are diagnostic
prompts, not a quality benchmark. The original 0.8B model itself makes mistakes
on some of them. Native validation was run separately for both backends.
For 0.8B, XNNPACK passes four of six smoke prompts and MLX passes five: both
mistranslate
"cat" to Polish, and XNNPACK also answers `17 * 23` incorrectly. The original
Hugging Face model also fails these two prompts; the exports are not claiming
that quantization preserves every greedy answer.
For 2B, both backends return the expected content in all six prompts. Both wrap
the JSON answer in a Markdown code fence despite the prompt requesting JSON only.

The short 2B greedy-decode check measured about 10.7 tokens/s on XNNPACK and
1.2 tokens/s on MLX with this Mac runtime; it is not a general throughput
benchmark. At 2,048 tokens the planned workspace is about 747 MiB for XNNPACK
and 651 MiB for MLX, beyond the quantized weights. These are workspace sizes,
not peak app memory measurements. Larger context lengths have not been tested.
