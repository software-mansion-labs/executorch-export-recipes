# gemma4_omni_mlx notes

Gemma 4 E2B omnimodal (vision + audio + text) MLX export, onboarded from the
executorch repo's `export_gemma4_omni_mlx_v3.py`. Published contract:
`software-mansion/react-native-executorch-gemma-4-multimodal` (`e2b/mlx/config.json`,
4 methods: token_embedding/text_decoder/vision_encoder on MLX, audio_encoder on XNNPACK).

Structural-only in this harness: backend=mlx + text-generation → not runtime-verifiable,
and the real export needs an executorch source build with the MLX backend. The published
.pte already exists on HF.

## Export settings

`group_size=64` (the v3 recipe's CLI default is 32, but the published multimodal artifact
was produced at gs=64 — same on-device-tested setting as the text `gemma4_mlx`). int4
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
