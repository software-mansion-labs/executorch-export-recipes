# gemma4_omni notes

Gemma 4 E2B omnimodal (vision + audio + text) MLX export, onboarded from the
executorch repo's `export_gemma4_omni_mlx_v3.py`. Published contract:
`software-mansion/react-native-executorch-gemma-4-multimodal` (`e2b/mlx/config.json`,
4 methods: token_embedding/text_decoder/vision_encoder on MLX, audio_encoder on XNNPACK).

The published config declares the canonical `image-text-to-text` and
`audio-text-to-text` capabilities (these are in the spec enum so the spec loads).
Producing the `.pte` needs an executorch source build with the MLX backend; the
published `.pte` already exists on HF.

## Export settings

`group_size=64` (the v3 recipe's CLI default is 32, but the published multimodal artifact
was produced at gs=64 — same on-device-tested setting as the text `gemma4`). int4
weights; `max_seq_len=2048` (the published model's context). These are baked into
`export.py` to match the shipped `.pte`.

## Self-containment

The full gemma4 model + export code is vendored under `_gemma4/` (the encoders,
omni v2/v3 builders, speech_transform, quant_utils, and the text_decoder package).
Cross-imports were rewritten from `executorch.examples.models.gemma4.X` to relative
(`from .X import ...`), so there is NO `executorch.examples.models.gemma4` coupling.

The only irreducible executorch dependencies are the compiled backends, resolved at
export time in an ET build:
- `executorch.backends.mlx.*` (MLXPartitioner, quantize_model_, passes, custom_ops, llm.cache)
- `executorch.backends.xnnpack.*` (audio_encoder partitioner)
- `executorch.extension.llm.custom_ops`, `executorch.exir.*`

## Known residual (acceptable)

`_gemma4/quant_utils.py:apply_embedding_quantization` imports
`executorch.examples.models.llama.source_transformation.quantize.EmbeddingQuantHandler`.
This is a LAZY import inside that function, and the function is only called on
`export_gemma4.py`'s XNNPACK embedding-quant path — NOT the omni MLX export path
(v3 uses `executorch.backends.mlx.llm.quantization.quantize_model_`). So it is dead
code for this model, does not affect import-safety, and was left verbatim rather than
editing vendored model logic. If a fully example-decoupled vendor is ever wanted,
inline EmbeddingQuantHandler or drop the unused embedding-quant path.

## Core ML: not supported, and not attempted

This model has **no Core ML export and is not expected to get one.** Apple-silicon
acceleration here is MLX only (see `mlx/export.py`), which additionally requires the
software-mansion-labs ExecuTorch fork -- the PyPI wheel produces an MLX payload the
RNE runtime cannot load.

**Why Core ML is not on the table.** This is an autoregressive decoder driven by a
mutable KV cache. ExecuTorch's Core ML backend has no practical path for that: the
cache is a mutable buffer carried across calls, and the decode step needs a growing
sequence dim. Core ML's dynamic-shape support is the exact area where we already
have a confirmed, filed defect -- non-CPU compute paths reject shapes Core ML infers
as data-dependent (apple/coremltools#2825, see `docs/coremltools-2825-repro.md`).

**Status: a judgment call, not a measured result.** Nobody has run this model through
the Core ML backend and collected a failure. It is not attempted because the cost is
high and the prior is bad, not because it was tried and rejected. If Core ML gains
usable stateful-KV-cache support, this is worth revisiting.
