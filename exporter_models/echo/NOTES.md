# echo

Echo (`software-mansion/Echo`, private) is a zero-shot voice-cloning TTS pruned
and distilled from OmniVoice: a bidirectional 24-layer Qwen3 that fills masked
Higgs Audio v2 codec tokens (8 codebooks x 1024 codes, 25 frames/s, 24 kHz).
The export fetches the checkpoints (`pretrain/{fp32,w4a16,w8a16}`) at a pinned
revision with the cached Hugging Face login, and the codec from
`eustlb/higgs-audio-v2-tokenizer` (`_echo/hub.py`). `ECHO_LOCAL_DIR` points it
at a local copy laid out like the repo, with the codec in `codec/`.

The sampler (8 main passes + 2 polish passes) stays in the caller; the
reference is `run/voice_cloning.ipynb` in the source repo, ported in
`evaluate.py`.

## Files

| backend | file | what |
| --- | --- | --- |
| xnnpack | `echo_xnnpack_fp32.pte` | backbone, fp32 |
| xnnpack | `echo_xnnpack_8da4w.pte` | backbone, w4a16 codes + int8 dynamic activations |
| xnnpack | `codec_decoder_xnnpack_fp32.pte` | codec decoder, T <= 1024 |
| xnnpack | `codec_encoder_xnnpack_fp32.pte` | codec encoder, 10 s reference |
| vulkan | `echo_vulkan_fp16.pte` | backbone, fp16 |
| vulkan | `echo_vulkan_8da4w.pte` | backbone, w4a16 codes + int8 dynamic activations |
| mlx | `echo_mlx_fp16.pte` | backbone, fp16 |
| mlx | `codec_decoder_mlx_fp32.pte` | codec decoder, T <= 1024 |
| coreml | `echo_coreml_fp16.pte` | backbone, fp16 |
| coreml | `codec_decoder_coreml_fp16.pte` | codec decoder, static T = 250 |

Android pairs the Vulkan backbone with the XNNPACK codec files: Vulkan cannot
hold the decoder in one blob above T = 16 and is 8x slower than XNNPACK on it.
The encoder is XNNPACK only (80 blobs, exact against `codec.encode`).

## Contract

```
encode_prefix(text i64[P], audio i64[8, P], is_audio f32[P], pos i64[P])
    -> k_0, v_0, ..., k_23, v_23           48 x f32[6, P, 128]
denoise(codes i64[8, T], pos i64[T], bias f32[P], k_0, v_0, ..., k_23, v_23)
    -> log_probs f32[8, T, 1024]
decode(codes i64[8, T]) -> audio f32[T * 960]
encode(wav f32[240000]) -> codes i64[8, 250]
```

P <= 768, T <= 1024 (backbone). The prefix is text ids plus reference codes
as one sequence blended by `is_audio` (torch.export cannot declare a dim as a
sum of two). `pos` is passed in; the target continues at P. `bias` adds to
every target score towards a prefix key: 0 on the main passes, +1 on the
reference frames for the polish passes. The prefix K/V is computed once per
clip and fed back into every `denoise`. Float I/O is fp32 on every backend.

## Why the odd spellings

- **fp16 overflow.** The residual stream reaches |x| ~ 1.1e4 (mean(x^2) ~
  1.8e5), so a decomposed RMSNorm overflows fp16 (0.3% argmax agreement vs
  99.0% with the fused op). `RMSNorm.mode` picks the spelling per backend:
  `native` (`aten.rms_norm`) for MLX, Core ML and QNN; `vulkan`
  (x * rsqrt(mean(x * x) + eps), which Vulkan fuses into its fp32-accumulating
  kernel); `linear` (mean as a constant linear) for XNNPACK fp32.
- **XNNPACK one blob per method.** Reshapes use a literal -1 (a symbolic size
  is rejected), reshape after transpose is `.contiguous().view()`, K/V travel as
  48 tensors (no dynamic slicing), RMSNorm's mean and RoPE's rotate_half are
  constant linears, attention is bmm/softmax/bmm (SDPA's masked-row guard is
  not delegated). Codec Snake activations get 1/alpha baked in, otherwise each
  of the 36 splits the delegate.
- **QAT code injection.** The quantized checkpoints' codes are the model.
  torchao would re-derive scales as amax / 7.5 (int4) and amax / 127.5 (int8),
  but the w4a16 codes stop at +-7 and only 58% of groups reach it, so
  `_echo/quant.py` overwrites torchao's qdata/scale with the checkpoint's.
- **Core ML eps.** RMSNorm's eps is a [1] tensor: coremltools' fuse_linear_bias
  crashes on a 0-d bias (apple/coremltools#2911).
- **Core ML codec is static** (T = 250, 10 s); the caller pads codes and trims audio.

## Environment

XNNPACK and Core ML run in this dir's uv project (released executorch 1.4.1).
MLX and Vulkan need the software-mansion-labs fork
(`@ms/separate-backends-1.4.1`) and run from its source build
(`skills/backends/vulkan`). Exporting the shipped Vulkan graphs needs only the
fork base: fork PRs #25 (int4 weight-only linear after a view,
pytorch/executorch#24116) and #26 (`aten.rms_norm` decomposition,
pytorch/executorch#24119) leave both files byte-identical. The runtime needs
fork #27 (dq8ca linear wrong when M > K): without it `echo_vulkan_8da4w.pte`
breaks on prompts with P > 512 (Polish WER 987% -> 0% with it).

## QNN (standalone, not in the harness)

`qnn/export_qnn.py` lowers both methods at one static bucket (P = 320, T = 256)
for SM8850 with HTP weight sharing; `qnn/calibrate.py` builds the a16w8
calibration set from the source repo's dev set. Needs QAIRT 2.47 and a box
with enough RAM for the full model. These scripts were consolidated from the ones that
built the measured QNN files and were not re-run after that (the calibration set they
produce is identical).

```
python qnn/export_qnn.py echo_qnn_fp16_p320_t256.pte --precision fp16
python qnn/calibrate.py calib_p320_t256.pt
python qnn/export_qnn.py echo_qnn_a16w8_p320_t256.pte --precision a16w8 --calib calib_p320_t256.pt
```

Differences from the dynamic contract: ids and positions are int32 (no int64
on the HTP); padding is masked with additive -1e4 (`key_bias` [P] on
`encode_prefix`, `self_bias` [T] on `denoise`, padded entries of `bias`).
a16w8 uses the w8a16 checkpoint and also returns the last hidden state:
otherwise the last layer's RoPE'd key only feeds a graph output and the HTP
reads its gathered cos/sin wrongly (pytorch/executorch#24125).

## Results

Whisper-small WER on the source repo's 14 test clips (`evaluate.py`):

| build | device | WER |
| --- | --- | --- |
| reference fp32 | host | 9.50% |
| XNNPACK fp32 | host | 9.50% (bit-identical codec tokens) |
| 8da4w recipe, torchao eager | host | 8.14% |
| XNNPACK 8da4w | host | 9.50% |
| Vulkan fp16 | Galaxy S26 Ultra | 8.60% |
| Vulkan 8da4w (runtime with fork #27) | Galaxy S26 Ultra | 10.41% |
| QNN fp16, 6 clips that fit P = 320 | Galaxy S26 Ultra | 10.1% (reference 6.06%) |
| QNN a16w8, same 6 clips | Galaxy S26 Ultra | 8.08% |

iPhone 16, warm, ms per call:

| backend | encode_prefix | denoise | codec decode |
| --- | --- | --- | --- |
| MLX fp16 | 135 | 132 | 479 |
| Core ML fp16 | 156 | 137 | 231 (T = 250) |

Both match fp32 (argmax agreement ~0.99). Core ML compiles on first load
(codec 69 s, backbone ~20 s), cached afterwards. Core ML fp16 `log_softmax`
underflows to -inf for probabilities below 1e-38 (codebooks 0-1): harmless for
argmax, clamp before any guidance arithmetic.

## Reproduce

```
uv run --project exporter_models/echo exporter export echo --backend xnnpack --output-dir <dir>
uv run --project exporter_models/echo exporter export echo --backend coreml --output-dir <dir>
PYTHONPATH=src <fork venv>/bin/python -m exporter.cli export echo --backend mlx --output-dir <dir>
PYTHONPATH=src <fork venv>/bin/python -m exporter.cli export echo --backend vulkan --output-dir <dir>
```

`multi_pte`: each command writes every file of that backend (`--local-spec`
with a one-variant config exports one). XNNPACK and Vulkan output is
deterministic; MLX differs only in a 4-hex prefix of its named-data keys,
and Core ML embeds a random model id, so neither repeats byte for byte.
