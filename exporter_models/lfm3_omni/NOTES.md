# LFM3 Omni 2B

XNNPACK export and native simulator numerical checks pass. The corrected MLX
int4 export also passes component parity, text, speech synthesis and mixed-output
smoke checks on an iPhone SE (3rd generation). See the limits below; these checks
are not a broad quality evaluation or performance benchmark.

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
For MLX, the inverse DFT uses matrix multiplication plus four-hop overlap-add
instead of transposed convolution. Explicit floor operations reconstruct the FSQ
digits. Codec weights stay in original fp32, without an intermediate bf16 cast.
Vision/audio towers and speech heads are unquantized;
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

## Exported artifacts and exact native runtime

| Artifact | Bytes | SHA256 |
| --- | --- | --- |
| XNNPACK 8da4w | 3248028928 | `aefbfd0fc40eed0f1b2951371e4343c49f11f145ddafaa70259691eb0d026741` |
| MLX int4 | 1951492992 | `12ce4d818a77c3ed7c098cc5b5ebd6612107863f599949b04740e58c1ca7de9d` |

A native arm64 iOS simulator probe linked the **same** ExecutorchLib,
XnnpackBackend and threadpool binaries packaged by Gallery's pinned library,
commit `4b7d6fd6fea10f9b3422121d6e439ccf88df93b2`. All seven methods executed.
The standard Mac pybinding omits `quantized_decomposed::embedding_byte.out`,
which the shipping simulator binary includes; its failure to load the embedding
method does not establish a failure in the shipping runtime.

Against the eager wrappers, native depth maximum absolute error was 0.0000244,
codec 1/4/32-frame error <= 0.00000274, and audio input 400/16400/80400-sample
error <= 0.00000473. All passed the predeclared component budget above.
A fixed 512-pixel image's native vision output also matched the original upstream
pipeline (maximum absolute error 0.0000294, cosine >= 0.99999).

Quantized text was compared with the original upstream fp32 Lfm3Model on a saved
31-token prompt, before generation. The acceptance budget, set before measurement,
was cosine >= 0.98, normalized RMSE <= 0.10, top-10 token overlap >= 0.5, and
matching greedy first token. Results: cosine **0.997907**, normalized RMSE
**0.066048**, top-10 overlap **0.9**, maximum absolute logit error **3.86805**,
and matching first token **28**. Native greedy generation answered the fixed
question “What is 2 + 2? Answer briefly.” with **4**, then the end-of-turn token.
This is a narrow regression check, not a broad quality evaluation or benchmark.

Run `validate.py --source SNAPSHOT --reference --output REPORT.json` in the
transformers 5.20.2 reference environment. It checks the speech and audio
reimplementations against upstream. In a compatible native host environment,
use `--pte MODEL.pte` instead of `--reference` to check a loaded artifact.
`--native-dir DIR` compares saved simulator component outputs for the same seed-42
fixtures. `validate_text.py --source SNAPSHOT --native-dir DIR --output REPORT.json`
compares saved `prompt.bin` (int64 IDs) and `prefill-logits.bin` (float32) with the
original upstream text model. Run these checks separately from exports.

Initialization detail: the export explicitly marks `conv_state` buffers with
`InitializedMutableBufferPass`, preserving zero short-convolution histories for
fresh loads. Unwritten XNNPACK KV rows are masked or excluded before they become
visible. A fresh model per independent turn avoids relying on unsupported cache
reset semantics across methods.

Delegation: MLX lowers each of the seven methods to one delegate with no host
operators in the lowered graph. XNNPACK delegate counts were vision 56, audio 101,
text 350, depth 36, codec 75; token/audio embeddings remain portable operators.
These are delegation observations, not speed measurements.

## Physical MLX recovery and validation (2026-11-10)

Target: iPhone SE (3rd generation), A15, iOS 27.0.1. Gallery uses React Native
ExecuTorch commit `4b7d6fd6fea10f9b3422121d6e439ccf88df93b2` and its bundled
native binaries. No runtime or React Native library modifications were made.
Exports use the pinned Software Mansion ExecuTorch fork above, sequentially.

Two codec failures were reproduced and isolated:

- Inverse-DFT transposed convolution attempted to allocate 18,903,859,200 bytes
  for one frame on the available Mac MLX runtime. The physical decoder also
  failed with `Error::Internal`. The model-owned matrix multiply/overlap-add
  decomposition removes that temporary without changing waveform math.
- Intermediate probes located an FSQ mismatch before the first linear layer:
  integer floor division was lowered as true division. Explicit fp32 floor
  operations recover exact mixed-radix digits for valid codes 0..2015.

The MLX path matches the original eager codec exactly on fixed 1/4/32/256-frame
fixtures. On the **full corrected artifact**, physical fp32 codec outputs pass the
predeclared max-absolute-error budget of 0.0005:

| Frames | Max absolute error | Cosine |
| --- | --- | --- |
| 1 | 1.89e-7 | >0.9999999999 |
| 4 | 2.62e-6 | >0.9999999999 |
| 32 | 4.71e-6 | >0.9999999999 |
| 256 | 6.98e-5 | >0.9999999996 |

Physical vision, depth, audio embedding, and audio encoder (400/16400/80400
samples) pass against source-validated fp32 native fixtures with the declared
bf16 budget: normalized RMSE <=0.05 and cosine >=0.995. Observed normalized RMSE
ranges from 0.00357 to 0.02577; minimum cosine is 0.99967.

The actual Gallery worklet produced `4` for the fixed arithmetic prompt,
0.8 seconds of audio for `Hello.`, and `Hello! How can I help you today?` plus
3.84 seconds of audio for mixed output. Outputs were finite and terminated
normally. Native playback completed. An audio-input conversational smoke check
also completed; this is not an ASR accuracy measurement.

Corrected full MLX int4 SHA256:
`43488c93c3dd7669c977ddd15fc68d1e2a1dbf28a67397c1d3190ba2e53c5ee6`.
Earlier MLX artifacts with the codec failures must not be published.

Use `validate.py --mlx-codec` with the existing reference/native modes to select
the MLX codec decomposition. Weights, exported programs and diagnostic fixtures
remain outside Git.

## Remaining limits

The available Mac MLX runtime uses an older generated wire mapping. The shipping
artifact therefore fails host text decoding with `scatter_add_axis` reporting a
rank/axis mismatch. Inspection confirms that the exported text graph has 180
QuantizedMatmul instructions (tag 122) and no ScatterAdd instruction; that older
host interprets tag 122 as ScatterAdd. A separate local diagnostic copy remapped
223 instruction-type bytes across the seven graphs to the older mapping, preserving
all weights and other graph fields. It then generated **4** and EOS for the same prompt.
That diagnostic established a Mac host wire mismatch. The corrected full export
has since been tested directly on iOS as described above. No runtime or React Native
library code was patched. The earlier artifact is retained separately for reproduction. The exporter mapping has CumMax=112, ArgPartition=121,
QuantizedMatmul=122 and ScatterAdd=123. Do not replace or patch the React Native
library to bypass this boundary. Physical iOS MLX validation also requires a
physical device; the simulator has no MLX delegate.

No physical-device speed, latency, network-disabled behavior or live
microphone/camera capture is claimed. Broad text/speech quality evaluation, long-context/window-boundary
checks and quantized held-out evaluation remain outside the completed smoke checks.
The MLX int8 recipe is present but was not exported or validated in this run.

## Reproducing native simulator checks

`native_probe.cpp` links the Gallery library's existing native binaries, so it
checks its operator coverage instead of the more limited Python wheel. This does
not change or rebuild the React Native library. With its npm package installed:

```sh
RNE_LIBRARY=/absolute/path/to/node_modules/react-native-executorch
RNE_SIM_FRAMEWORK="$RNE_LIBRARY/third-party/ios/ExecutorchLib.xcframework/ios-arm64-simulator"
xcrun --sdk iphonesimulator clang++ -std=c++20 -O1 -target arm64-apple-ios17.0-simulator \
  -isysroot "$(xcrun --sdk iphonesimulator --show-sdk-path)" \
  -I"$RNE_LIBRARY/third-party/include" -F"$RNE_SIM_FRAMEWORK" \
  exporter_models/lfm3_omni/native_probe.cpp -framework ExecutorchLib \
  -framework Foundation -framework Accelerate \
  -Wl,-force_load,"$RNE_LIBRARY/third-party/ios/XnnpackBackend.xcframework/ios-arm64-simulator/libXnnpackBackend.a" \
  "$RNE_LIBRARY/third-party/ios/libs/executorch/libthreadpool_simulator.a" \
  -Wl,-rpath,"$RNE_SIM_FRAMEWORK" -o /absolute/path/to/native-probe
python exporter_models/lfm3_omni/write_fixtures.py --output /absolute/path/to/fixtures --tokenizer SNAPSHOT/tokenizer.json
xcrun simctl spawn SIMULATOR_UDID /absolute/path/to/native-probe MODEL.pte /absolute/path/to/fixtures/text
xcrun simctl spawn SIMULATOR_UDID /absolute/path/to/native-probe MODEL.pte /absolute/path/to/fixtures/components components
python exporter_models/lfm3_omni/validate.py --source SNAPSHOT --native-dir /absolute/path/to/fixtures/components --output REPORT.json
```

All paths passed to the simulator process must be absolute. The optional `vision`
probe mode consumes `image.bin` as float32 contiguous `[1,3,512,512]` RGB 0..255
and saves `actual-vision.bin`. Keep fixtures and probe/build outputs outside Git.
