# LFM3 Omni 2B

Implementation checkpoint. Final `.pte` runtime verification is still pending.
Do not treat the checks below as verification of exported or on-device inference.

## Provenance

- Checkpoint: `LiquidAI/LFM3-Omni-2B`.
- Immutable source revision: `21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`.
- Original remote model code: HuggingFace, Apache-2.0. Architecture adaptations
  retain that attribution; see `LICENSE.code`.
- Weights: LFM Open License v1.0, retained in `LICENSE.weights` and copied into
  each output directory. This recipe does not publish or relicense the weights.
- The text, vision and audio implementations are model-owned adaptations of
  this repository's LFM3 recipes. No cross-model imports are required.

## Contract

The artifact has seven methods. It is intended for custom low-level orchestration,
not the standard text-only LLM runner.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `vision_encoder` | f32 `[1,3,512,512]`, RGB 0..255 | f32 `[1,256,2048]` |
| `audio_encoder` | f32 `[1,400+160k]`, mono 16 kHz, 0 <= k <= 2997 | f32 `[1,ceil((k+1)/8),2048]` |
| `token_embedding` | i64 `[1,S]`, 1 <= S <= 256 | f32 `[1,S,2048]` |
| `text_decoder` | f32 `[1,S,2048]`, i64 `[S]` positions | f32 `[1,128000]` logits, f32 `[1,2048]` final hidden state |
| `depth_decoder` | f32 `[1,2048]`, i64 `[1,12]` codes, bool `[1,12]` masked | f32 `[1,12,2017]` logits |
| `audio_embedding` | i64 `[1,12]` codes | f32 `[1,1,2048]` |
| `audio_decoder` | i64 `[1,12,F]`, 1 <= F <= 256 | f32 `[1,F*1920]`, mono 24 kHz |

The text trunk owns mutable KV/short-convolution buffers. Load a fresh model for
independent requests. Prompt chunks must use consecutive positions. The exported
context limit is 4096. Input audio is right-padded to 400 + 160k samples.
Images are stretched to the fixed export size; this differs from upstream's
variable-resolution preprocessing. Video is not exposed by the Gallery screen.

Speech generation follows the upstream 4/4/4 MaskGIT schedule over a random order
of 12 codebooks. Any codebook emitting 2016 terminates a speech block; the complete
EOS frame is fed back into the language model but excluded from codec synthesis.
Interleaving alternates six text slots and twelve audio slots, with a final audio
tail following `<|text_end|>`. Audio/image placeholders belong inside the upstream
vision start/end markers. Thinking is not enabled.

The codec is decoded offline. EXPORT PATCH: inverse real FFT and overlap-add are
expressed as a fixed real-valued inverse-DFT transposed convolution and envelope
normalization. This preserves waveform semantics without complex tensor operators.
The codec is kept fp32. Vision/audio towers and speech heads are unquantized;
XNNPACK quantizes the language trunk with 8da4w, MLX with 4w or 8w.

## Environment and export

Python 3.10, stable torch 2.14.0, torchao 0.18.0, safetensors 0.8.0.
`pyproject.toml` and `uv.lock` declare the runtime-compatible ExecuTorch fork at
`0a0871900c8f63a032981ac4748b9cdda84bfcb8`. As with the other MLX recipes,
use a source-built environment; uv cannot build this fork directly because its
CMake setup requires the source directory to be named `executorch`.

The installed PyPI MLX schema is incompatible with the shipping runtime.
`require_mlx_runtime_fork()` deliberately rejects it. In an isolated export
environment, use the pinned fork's MLX backend and regenerate its serialization
bindings using `python backends/mlx/serialization/generate.py` with `flatc` installed.
Do not mutate another model's environment. Native host validation must also use
a runtime built from the matching fork. See `skills/backends/mlx` and
`skills/backends/vulkan` for the source-build setup.

From that environment, at the repository root:

```sh
PYTHONPATH=src python -m exporter.cli export lfm3_omni --size 2b --backend xnnpack --output-dir exports/lfm3-omni/xnnpack
# Run exports sequentially, especially on a 16 GB host.
PYTHONPATH=src python -m exporter.cli export lfm3_omni --size 2b --backend mlx --output-dir exports/lfm3-omni/mlx
```

`LFM3_OMNI_SOURCE=/absolute/path/to/snapshot` uses an already-downloaded pinned
snapshot, including `audio_tokenizer/model.safetensors`. Otherwise the recipe
fetches the pinned revision. The tokenizer, chat template, license and provenance
are written alongside the artifact. The int8 MLX variant is optional, selected
with the harness variant flag.

## Validation so far

Host: arm64 macOS 26.5.2. Original component reference: transformers 5.20.2,
fp32 eager inference. Fixed seed 42 for speech/audio, 17 for vision. These are
component parity checks using synthetic inputs, not task-quality evaluation.
Acceptance was cosine >= 0.99999 and maximum absolute error <= 0.0005.

| Component | Inputs | Maximum absolute error | Result |
| --- | --- | --- | --- |
| Speech depth decoder | mixed known/masked codebooks | 0.0000244 | pass |
| LiquidCodec | 1 / 4 / 32 frames | 0.000000207 / 0.000000485 / 0.00000302 | pass |
| Audio encoder + projector | 400 / 16400 / 80400 waveform samples | 0 / 0.000000537 / 0.00000120 | pass |
| Vision tower + projector | fixed byte-valued RGB image, matching upstream patchification | 0.0000334 | pass |

Pending: text-trunk comparison, actual artifact method metadata and hashes,
quantized task quality, loaded-runtime parity, delegation report, XNNPACK simulator
inference and Gallery lifecycle checks. MLX iOS delegate validation requires a
physical device; the iOS simulator has no MLX delegate. No device performance claim
is made. All seven method graphs were captured for both backends; lowering and
serialization have not yet been verified.
