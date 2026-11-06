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
| mlx | `echo_mlx_fp16.pte` | backbone, fp16, v2 contract (prefill / step / polish) |
| mlx | `codec_decoder_mlx_fp32.pte` | codec decoder, T <= 1024 |
| coreml | `echo_coreml_fp16.pte` | backbone, fp16, dynamic (GPU) |
| coreml | `echo_coreml_fp16_ane_t128.pte`, `_ane_t256.pte` | backbone, fp16, static bucket P = 320, T = 128 / 256, CPU_AND_NE |
| coreml | `codec_decoder_coreml_fp16.pte` | codec decoder, static T = 250, CPU_AND_GPU |

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

MLX uses the v2 contract (`_echo/v2.py`):

```
prefill(ids i64[NT], ref i64[8, R]) -> K, V  f16[24, 1, 6, P, 128]   (P = NT + R)
step(tokens i64[8, T], K, V) -> confidence f32[2, T], prediction i64[2, T]
polish(tokens i64[8, T], bias f16[1, 1, 1, P], K, V) -> tokens i64[6, T]
```

`step` is one main pass reduced in-graph to codebooks 0-1 (best code and its
log-probability); `polish` runs both polish passes and returns codebooks 2-7.
Two stacked fp16 caches and [2, T] outputs replace 48 fp32 tensors and
[8, T, 1024] per pass; Q/K/V and gate/up are fused, attention is SDPA and RoPE
is `mlx::rope`. The ANE buckets keep the v1 methods with `key_bias` [P] /
`self_bias` [T] (-1e4 on padding), like QNN.

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

`qnn/export_qnn.py` lowers the backbone, `qnn/export_codec_qnn.py` the codec
decoder, both for SM8850 (QAIRT 2.47); `qnn/calibrate.py` builds the backbone
a16w8 calibration set. Each script's docstring has its contract and the HTP
workarounds. Needs a box with enough RAM for one backbone export at a time.

```
python qnn/export_qnn.py echo_qnn_fp16_t256.pte --precision fp16          # P = 320 + 192, fp16 KV
python qnn/export_codec_qnn.py echo_codec_qnn_a16w8_t256.pte --precision a16w8
python qnn/calibrate.py calib_p320_t256.pt
python qnn/export_qnn.py echo_qnn_a16w8_p320_t256.pte --precision a16w8 --calib calib_p320_t256.pt
```

Differences from the dynamic contract: ids and positions are int32 (no int64
on the HTP); padding is masked with additive -1e4 (`key_bias` [P] on
`encode_prefix`, `self_bias` [T] on `denoise`, padded entries of `bias`).
fp16 splits denoise into `denoise` (argmax + log-prob of codebooks 0-1, main
passes) and `denoise_full` (raw logits, polish passes) and passes KV as fp16.
a16w8 uses the w8a16 checkpoint and also returns the last hidden state:
otherwise the last layer's RoPE'd key only feeds a graph output and the HTP
reads its gathered cos/sin wrongly (pytorch/executorch#24125).

Galaxy S26 Ultra, 10 s of speech (184-token prompt, 250 frames), fp16 backbone
+ a16w8 codec: prefix 26 ms + 10 passes 271 ms + codec 45 ms = 342 ms (29x
real time), Whisper-exact. fp16 codec: 80 ms, 53 dB to fp32.

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

Model time (prefill + 10 passes, sampler included), 123-frame reference,
125 / 250 target frames, median of 5 warm runs, ms:

| build | iPhone 16 (A18) | iPhone 17 (A19) | M3 Pro |
| --- | --- | --- | --- |
| MLX fp16, v2 contract | 698 / 1262 | | 278 / 473 |
| MLX fp16, v1 contract | 821 / 1580 | 293 / 530 | 340 / 560 |
| MLX int4 (w4a16), v1 | | 589 / 917 | |
| Core ML ANE, t128 / t256 | 695 / 865 | 596 / 814 | |
| XNNPACK 8da4w, v1 | 3059 / 6194 | | |

The ANE buckets win on A18 at 10 s; A19's GPU is ~3x faster on fp16 while
its ANE gains ~10%, so MLX fp16 wins there and quantization loses (int8 is 2x
slower than fp16 on A19, equal on A18). Codec decode on iPhone 16, Core ML
T = 250: CPU_AND_GPU 207 ms, CPU_ONLY 186, ALL 250, CPU_AND_NE 1090.

iPhone 16, warm, ms per call (v1 contract):

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
