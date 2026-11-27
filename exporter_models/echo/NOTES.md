# echo

Echo (`software-mansion/Echo`, private) is a zero-shot voice-cloning TTS pruned
and distilled from OmniVoice: a bidirectional 24-layer Qwen3 that fills masked
Higgs Audio v2 codec tokens (8 codebooks x 1024 codes, 25 frames/s, 24 kHz).
It is released as `software-mansion/Reverb-TTS`. The export fetches its
checkpoints from there at tag v1.0.0 (`model/safetensors/{fp32,int8,int8a8,int4}`;
int8, int8a8 and int4 hold the QAT integer codes and their scales, which every
backend keeps as they are), the codec from `eustlb/higgs-audio-v2-tokenizer`,
and the codec's int8 decoder and encoder checkpoints from
`software-mansion/higgs-audio-v2-tokenizer-ExecuTorch` (`decoder/safetensors/int8`, `encoder/safetensors/int8`),
all public, pinned in `_echo/hub.py`. `REVERB_LOCAL_DIR` points it at a local
copy of Reverb-TTS. Only `evaluate.py` and `qnn/calibrate.py` read the private
source repo (its test and dev sets), with the cached Hugging Face login.

The sampler (8 main passes + 2 polish passes) stays in the caller; the
reference is `run/voice_cloning.ipynb` in the source repo, ported in
`evaluate.py`.

## Files

| backend | file | what |
| --- | --- | --- |
| xnnpack | `reverb_xnnpack_8da8w.pte` | backbone, W8A8 checkpoint (int8 weights, int8 per-token inputs), fp16 attention and cache, v3 contract (`xnnpack/backbone.py`) |
| xnnpack | `codec_decoder_xnnpack_fp16.pte` | codec decoder, fp16 up to the fp32 tail, T 4-512 (`xnnpack/decoder.py`) |
| xnnpack | `codec_encoder_xnnpack_int8.pte` | codec encoder, int8 QAT linears in HuBERT, T 25-500 frames (`xnnpack/encoder.py`) |
| vulkan | `echo_vulkan_fp16.pte` | backbone, fp16, v3 contract (prefill / step / polish) |
| mlx | `reverb_mlx_fp16.pte` | backbone, fp16, v3 contract (prefill / step / polish) |
| mlx | `reverb_mlx_int8.pte` | backbone, w8a16 codes (int8 per channel), fp16 compute, v3 contract |
| mlx | `reverb_mlx_int4.pte` | backbone, w4a16 codes (int4, group 32), fp16 compute, v3 contract |
| coreml | `echo_coreml_fp16.pte` | backbone, fp16, dynamic (GPU) |
| coreml | `echo_coreml_fp16_ane_t128.pte`, `_ane_t256.pte` | backbone, fp16, static bucket P = 320, T = 128 / 256, CPU_AND_NE |
| coreml | `codec_decoder_coreml_int8.pte` | codec decoder, int8 GPTQ + QAT weights (per block of 8 input channels), T <= 1024, CPU_AND_GPU, precompiled |
| coreml | `codec_decoder_coreml_fp16.pte` | codec decoder, fp16, T <= 1024, CPU_AND_GPU, precompiled |
| qnn | `reverb_qnn_int4_v{69,73,75,79,81}.pte` | backbone, int4 codes (blocks of 32), fp16 activations, one per SoC (standalone `qnn/export_reverb_qnn.sh`, see below) |

Android pairs the Vulkan backbone with the XNNPACK codec files: the shipped
Vulkan runtime cannot hold the decoder in one blob above T = 16 and is 8x slower
than XNNPACK on it. (2026-11-12: with buffer-backed, fused conv1d kernels added
to the patched runtime, a Vulkan decoder runs 3.7-5.1x faster than the XNNPACK
fp16 one on a Galaxy S24 FE and closer to fp32; not in the recipe yet, see the
source repo's `knowledge/inference/codec_export.md`.)
The xnnpack and mlx files are the ones published in `software-mansion/Reverb-TTS` (`model/executorch/`) and
`software-mansion/higgs-audio-v2-tokenizer-ExecuTorch` (`{decoder,encoder}/xnnpack/`); the xnnpack graphs and their
ExecuTorch lowering patches are described in the docstrings of `xnnpack/{backbone,decoder,encoder}.py`.
`android/` is the demo app (QNN or Vulkan backbone; see its README). Its Vulkan
path loads the v1 programs (`encode_prefix` / `denoise`, fp16 and 8da4w), which
this recipe wrote up to commit dd65b02; it now writes the v3 program.

## Contract

```
encode_prefix(text i64[P], audio i64[8, P], is_audio f32[P], pos i64[P])
    -> k_0, v_0, ..., k_23, v_23           48 x f32[6, P, 128]
denoise(codes i64[8, T], pos i64[T], bias f32[P], k_0, v_0, ..., k_23, v_23)
    -> log_probs f32[8, T, 1024]
decode(codes i64[8, T]) -> audio f32[T * 960]
encode(wav f32[240000]) -> codes i64[8, 250]
```

The Core ML codec decoder names its one method `forward`, react-native-executorch's convention for single-method
programs.

P <= 768, T <= 1024 (backbone). The prefix is text ids plus reference codes
as one sequence blended by `is_audio` (torch.export cannot declare a dim as a
sum of two). `pos` is passed in; the target continues at P. `bias` adds to
every target score towards a prefix key: 0 on the main passes, +1 on the
reference frames for the polish passes. The prefix K/V is computed once per
clip and fed back into every `denoise`. Float I/O is fp32 on every backend.

MLX and Vulkan use the v3 contract (`_echo/v3.py`):

```
prefill(ids i64[NT], ref i64[8, R]) -> cache f16[2, 24, 6, P, 128]   (keys and values, P = NT + R)
step(tokens i64[8, T], cache) -> confidence f32[2, T], prediction i64[2, T], held f32[2, T]
polish(tokens i64[8, T], reference_frames i64[1], cache) -> tokens i64[6, T]
```

T <= 1024; prefill takes L <= 512 and R <= 512, so P <= 1024 always fits step and polish.
The cache is opaque to the caller, passed back whole. `step` is one main pass
reduced in-graph to codebooks 0-1: the best code, its log-probability, and the
log-probability of the code each slot holds now (`held`, for remasking;
meaningless for masked slots). `polish` runs both polish passes and returns
codebooks 2-7: +1 on the scores towards the reference frames (the prefix's last
`reference_frames` positions), the second pass guided by 0.5 x (second -
first). 0.5 beats 1.0 by +0.001 SIM-o at the same WER on MiniMax
multilingual (source repo, `knowledge/benchmarks/minimax_multilingual.md`).
One fp16 cache and [2, T] outputs replace 48 fp32 tensors and
[8, T, 1024] per pass. The ANE buckets keep the v1 methods with `key_bias` [P] /
`self_bias` [T] (-1e4 on padding), like QNN.

What makes v3 fast, measured on an M4 Pro at T = 200 unless stated:

- **Q/K/V as one matmul**: a step from 31.4 to 28.4 ms in fp16 (-10%). Gate
  and up fused gain nothing (0.97-1.0x), so they stay apart.
- **Weights stored transposed**: `aten.linear` lowers to a transpose plus a
  matmul, `x @ W` to one node.
- **`mlx::rope` in every method, the sequence shifted by P**: the op takes only
  a symbolic offset and prefill's would be 0, so the prefix sits at [P, 2P) and
  the target from 2P (RoPE only sees relative positions; fp32 KL 1.2e-7).
- **No input copies** (`MemoryPlanningPass(alloc_graph_input=False)`): the K/V
  cache goes back in on every pass and is wrapped, not copied.
- **Constants as parameters, not buffers**: ExecuTorch drops the parameters a
  method doesn't use but keeps every buffer (fp16 program 693 -> 612 MB).
- **Audio heads in fp32**, each codebook's rows centered: the logits reach
  +-420, where fp16's step is 0.25 (fp16 heads: no faster, KL 1.9e-5 vs 1.3e-5).
  The held log-probability is a masked amax, as MLX has no gather.
- **The quantized checkpoints' own codes** go through `torchao::dequantize_affine`,
  which MLX fuses into its quantized matmul; the scales are cast to fp16, as a
  quantized matmul runs in the dtype of its scales.

Matmuls and attention are ~94% of a step; the fp16 matmuls run at ~70% of the
GPU's peak. Tried and dropped: bf16 (same speed, KL 8.0e-4 vs 1.3e-5), fp32
q/k norms, the sampler in the graph (the host sampler is 1.3% of the loop).

The Vulkan program (`VulkanEcho` and friends in `_echo/v3.py`) keeps that I/O,
so react-native-executorch's Reverb pipeline runs it unchanged:

- **Same cache shape, different data**: f16[2, 24, 6, P, 128] holding
  [2, 24, P, 6, 128], the [keys, heads, dim] layout `llama.custom_sdpa` reads.
  The caller never looks inside; the reshapes go through `flatten()` because
  torch.export guards a direct view.
- **Vulkan's own ops**: `et_vk.apply_rotary_emb_hf` (positions from 0, the
  target from P), `llama.custom_sdpa` with an additive [T, keys] mask (it is
  causal without one: zeros for prefill and steps, +1 on the reference frames
  for polish), `aten.rms_norm` (fp32 accumulation), weights as [out, in].
- **Spelled around missing ops or runtime bugs**: SiLU as `g * sigmoid(g)`;
  `held` by `gather` on the log-softmax (no `log`); the codebook sum with
  `keepdim`; `squeeze(2)`, not `squeeze(-1)` (the squeeze resize ignores
  negative dims); the polish bias from a flipped `arange` (its resize miscounts
  negative steps). `lower_vulkan` disables the fork's `causal_sdpa` pattern for
  `custom_sdpa`s without an LLM `update_cache`, on which it asserts.
- **Heads in fp32**, the main ones padded to 1028 rows per codebook (-1e4 bias)
  so the head matmul takes the vectorized kernel; the polish heads split into
  codebooks 2-3 and 4-7, so no rows are stored twice.
- One delegate per method; only the cache's `sym_size` runs outside.

## Why the odd spellings

- **fp16 overflow.** The residual stream reaches |x| ~ 1.1e4 (mean(x^2) ~
  1.8e5), so a decomposed RMSNorm overflows fp16 (0.3% argmax agreement vs
  99.0% with the fused op). `RMSNorm.mode` picks the spelling per backend:
  `native` (`aten.rms_norm`) for MLX, Core ML and QNN; `vulkan`
  (x * rsqrt(mean(x * x) + eps), which Vulkan fuses into its fp32-accumulating
  kernel); `linear` (mean as a constant linear) for the v1 XNNPACK fp32 program.
- **XNNPACK one blob per method (v1 program, no longer exported).** Reshapes use a literal -1 (a symbolic size
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
- **Core ML codec in fp16.** Snake's 1/alpha reaches 2.4e6 (the smallest
  |alpha| is 4.1e-7), past fp16's range, so the Core ML decoder computes
  x + sign(a) (sin(a x) / sqrt|a|)^2 instead (`Fp16Snake`; 950 alphas are
  negative).
- **Core ML codec int8 from the QAT checkpoint.** The weights are the source
  repo's int8 decoder (GPTQ, then QAT of the scales; `hub.decoder_int8()`), one
  scale per 8 input channels x every tap. That is Core ML's `per_block` 8 grid
  (symmetric [-127, 127], scale = block max / 127), and 99.7% of the blocks
  reach +-127, so the backend's quantizer (`op_linear_quantizer_config`) gets
  the checkpoint's integers back from the dequantized weights: the program is
  55.1 dB from the checkpoint decoded in fp32, the fp16-compute floor. Rounding
  the codec's own weights instead (round to nearest per block) gives 36.5 dB
  against fp32, the QAT weights 44.2 dB. The codebook tables stay fp16.
- **Core ML codec without the codebooks' training buffers** (`embed_avg`,
  `cluster_size`, `inited`): decode never reads them, but ExecuTorch keeps every
  buffer (2.1 MB per program).

## Environment

Core ML runs in this dir's uv project (released executorch 1.4.1). XNNPACK's
published files were reproduced byte for byte in the fork's venv (branch
`@is/vulkan-echo-int4-1.4.1` at 7764e753e, torch 2.13.0, torchao 0.18.0).
MLX and Vulkan need the software-mansion-labs fork
(`@ms/separate-backends-1.4.1`) and run from its source build
(`skills/backends/vulkan`); `echo_vulkan_fp16.pte` was exported at 0a087190.
Its runtime needs `llama.custom_sdpa` with an `attn_mask` (fork 9c1e9b63c):
react-native-executorch's native libs v0.11.1 (built from 0a087190) run it,
v0.11.0 aborts in `sdpa_impl` ("attention without an attn_mask must be
causal").

## QNN: what react-native-executorch ships (standalone, not in the harness)

`software-mansion/Reverb-TTS` `model/executorch/qnn/int4/reverb_qnn_int4_v{69,73,75,79,81}.pte` (v1.0.0) is the int4
backbone for SM8450 / SM8550 / SM8650 / SM8750 / SM8850, one file per SoC; react-native-executorch pairs it with the
Vulkan codec (`QNN_VULKAN_INT4`). `qnn/export_qnn_v3.py` writes it; its docstring has the interface and the HTP
workarounds.

- **Weights:** the int4 checkpoint's own codes and scales (blocks of 32), as the HTP's fp16-activation block
  quantization.
- **Methods:** `prefill`, then `step` at T 128 / 256 / 512 (`step_t128`, `step_t256`, `step`). Polish is two `step`
  calls (`polish_with_step`), never one graph: with int4 weights the HTP keeps every layer's fp16 expansion alive
  across the two passes (552 MB of spill-fill on SM8650, 23 MB for `step`).
- **I/O:** ids int32; the cache is 48 tensors f16[6, 1024, 128] (prefill's outputs, every step's inputs); inputs
  unplanned with `shared_buffer` on, outputs planned.

Environment: x86_64 Linux, QAIRT 2.47 (`QNN_SDK_ROOT`), Android NDK r26d, and the software-mansion-labs fork at
328163835 (`@ms/separate-backends-1.4.1`) with `qnn/patches/bw_float_block.patch` applied (int4 weights with fp16
activations, `QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK`, in the QNN ahead-of-time path), built with
`./install_executorch.sh --editable` and `./backends/qualcomm/scripts/build.sh --release`; transformers 5.14.1.

```
git -C $EXECUTORCH_ROOT apply $PWD/qnn/patches/bw_float_block.patch     # before building
source $QNN_SDK_ROOT/bin/envsetup.sh
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$EXECUTORCH_ROOT/build-x86/lib:$LD_LIBRARY_PATH
qnn/export_reverb_qnn.sh <out dir>                # all five; or: python qnn/export_qnn_v3.py out.pte --soc SM8650
python qnn/check_v3.py                            # eager check of the graphs against v3's arithmetic
```

One SoC at a time: each export peaks at ~21 GB of RAM and takes ~6 min on 24 cores. The QNN compile does not repeat
byte for byte (two exports of the same graph differ by ~20 KB); compare programs by their graph I/O and spill sizes
(`qnn-context-binary-utility`) or on a phone. OnePlus 12 (SM8650): 385 / 590 ms of program time per 5 s / 10 s
generation (prefill, 8 main steps, 2 polish steps); 739 MB of DMA-BUF.

## QNN v1 (standalone, not in the harness)

`qnn/export_qnn.py` lowers the backbone, `qnn/export_codec_qnn.py` the codec
decoder, both for SM8850; `qnn/calibrate.py` builds the backbone a16w8
calibration set. Each script's docstring has its contract and the HTP
workarounds. Environment: QAIRT 2.47 (`QNN_SDK_ROOT`) and the fork
`@ms/separate-backends-1.4.1` built with its QNN backend (verified at 328163835;
plus transformers 5.14.1 as pinned in `pyproject.toml`).
One backbone export at a time needs ~30 GB of RAM. The demo app (`android/`)
lists every file it loads and the command that makes it.

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
| Vulkan fp16, v1 | Galaxy S26 Ultra | 8.60% |
| Vulkan 8da4w, v1 (runtime with fork #27) | Galaxy S26 Ultra | 10.41% |
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
slower than fp16 on A19, equal on A18). The earlier static (T = 250) fp16
Core ML codec on iPhone 16: CPU_AND_GPU 207 ms, CPU_ONLY 186, ALL 250,
CPU_AND_NE 1090.

MLX v3, ms of model calls per generation (prefill, 8 main passes, polish; x
real time), references and targets of exactly 125 / 250 frames, median of 10
after 2, the mean of two rounds in opposite orders (60 s cool-down before
each program). M4 Pro: this fork's source build. iPhone 17 (A19, iOS 26.6):
react-native-executorch's native libs v0.11.1; v0.10.0's runtime rejects
these programs at load (InvalidProgram).

| reference + target | fp16 (612 MB) | int8 (352 MB) | int4 (232 MB) |
| --- | --- | --- | --- |
| M4 Pro, 5 s + 5 s | 197 (25.3x) | **179 (27.9x)** | 181 (27.6x) |
| M4 Pro, 5 s + 10 s | 317 (31.6x) | **300 (33.3x)** | 302 (33.1x) |
| M4 Pro, 10 s + 5 s | 219 (22.8x) | **200 (25.0x)** | 203 (24.7x) |
| M4 Pro, 10 s + 10 s | 345 (29.0x) | **329 (30.4x)** | 333 (30.1x) |
| iPhone 17, 5 s + 5 s | **223 (22.4x)** | 437 (11.4x) | 436 (11.5x) |
| iPhone 17, 5 s + 10 s | **384 (26.0x)** | 640 (15.6x) | 622 (16.1x) |
| iPhone 17, 10 s + 5 s | **268 (18.6x)** | 510 (9.8x) | 497 (10.1x) |
| iPhone 17, 10 s + 10 s | **460 (21.7x)** | 718 (13.9x) | 703 (14.2x) |

One step at P = 173, T = 250: M4 Pro 29.5 / 28.3 / 28.4 ms, iPhone 17 39.2 /
65.2 / 64.1 ms (fp16 / int8 / int4). int4's rounds differ by up to 5% on the
phone: it ran first (coolest) in one and last in the other.

Each matches its checkpoint run in fp32 PyTorch, teacher-forced on 8 test
generations: KL 1.3e-5 (fp16), 1.5e-5 (int8), 1.2e-5 (int4, measured before
its tables went int8), argmax agreement 99.6-99.9%. On the iPhone 17 int8 and
int4 tie and fp16 is 1.6-2x faster: MLX's matrix-unit kernels for A19's GPU
speed fp16 matmuls up more than quantized ones (likely, not profiled). On the
M4 Pro the quantized programs are 5-10% ahead.

Vulkan v3 (`echo_vulkan_fp16.pte`) on a Galaxy S24 FE (Exynos 2400e, Xclipse
940, Android 17), same setups, median of 5 after 2, 90 s cool-down before each
cell, one round. "Patched" is the fork with runtime changes (shared-memory fp16
matmuls with fp32 accumulation, round-to-nearest fp16, fp32 elementwise math
and attention accumulation) that are not in the fork; details and the profile
in the source repo's `knowledge/inference/vulkan_export.md`.

| reference + target | RN native libs v0.11.1 | patched runtime |
| --- | --- | --- |
| 5 s + 5 s | 1796 (2.8x) | **1114 (4.5x)** |
| 5 s + 10 s | 3083 (3.2x) | **1786 (5.6x)** |
| 10 s + 5 s | 2041 (2.4x) | **1308 (3.8x)** |
| 10 s + 10 s | 3922 (2.5x) | **2021 (4.9x)** |

Teacher-forced against the fp32 checkpoint (4 cases): argmax agreement 95.1%
(v0.11.1) and 99.5% (patched), polish 4-7 80.1% / 98.3%; MLX fp16 99.7% /
97.3%. Generated on the phone with react-native-executorch's sampler, both say
every word (faster-whisper large-v3, 0/68 word errors). 3.42 GB of GPU memory
with the three methods loaded (three copies of the weights, activations planned
for the bounds); 3.3-4.5 s to load.

Vulkan int4 (the w4a16 checkpoint, 233 MB; exported by a scratch script, not a
variant here yet, as `VulkanEcho` takes fp32 checkpoints only), against its own
checkpoint on 4 teacher-forcing cases (P 163-274, T 125 / 250), same phone:
v0.11.1 1204 / 2014 / 1497 / 2169 ms, 96.1% argmax agreement, 1.89 GB of GPU
memory; patched runtime 1014 / 1629 / 1156 / 1785 ms, 99.6% (MLX int4 99.7%),
0.61 GB. These need two partitioner options, which `lower_vulkan` now sets and
runtimes without the patch ignore: `shared_workspace`, so the three methods get
one set of working memory (they never run at once; packed weights are shared in
any case), and `fp16_round_to_nearest`, so fp16 results round to nearest even
(Samsung's Xclipse driver rounds toward zero by default).

Core ML codec decoder: SNR on 14 test clips (132 s) against the eager fp32
decode; ms per 5 / 10 s of audio, median of 20 after 3, the mean of two
passes, same devices and runtimes as above:

| program | size | SNR mean / min | M4 Pro | iPhone 17 | load, M4 Pro / iPhone 17 |
| --- | --- | --- | --- | --- | --- |
| `codec_decoder_coreml_int8.pte` | 23.3 MB | 44.2 / 39.8 dB | 25.8 / 49.1 | 52.6 / 108.8 | 0.92 / 0.87-0.96 s |
| `codec_decoder_coreml_fp16.pte` | 43.6 MB | 54.8 / 50.2 dB | 25.7 / 49.3 | 52.6 / 107.0 | 0.10-0.15 / 0.15-0.23 s |

Memory on the iPhone 17: the app's `phys_footprint` (what iOS counts against
its ~3.5 GB limit, GPU allocations included), one program per launch, MB:

| program | loaded | peak, typical (10 s + 10 s / 10 s) | peak, maximum (P = T = 1024 / 1024 frames) |
| --- | --- | --- | --- |
| `echo_mlx_fp16.pte` | 161 | 558 | 847 |
| `echo_mlx_int8.pte` | 193 | 587 | 877 |
| `echo_mlx_int4.pte` | 230 | 623 | 920 |
| `codec_decoder_coreml_fp16.pte` | 112-138 | 180-205 | 570-590 |
| `codec_decoder_coreml_int8.pte` | 623-629 | 689-696 | 1099-1106 |

On the phone the quantized programs save download size only: MLX int8 /
int4 take more memory than fp16, and the Core ML int8 codec ~500 MB more than
fp16 (Core ML holds it at ~610 MB resident from the load on).

Rounding the codec's own weights per block instead (no QAT) gives 36.5 dB at
the same size and speed. The codec's own loss, its fp32 decode against the
original audio, is ~3 dB log-mel L1. int8's first load after installing the
app took 5.0 s on the phone.

iPhone 16, warm, ms per call (v1 contract):

| backend | encode_prefix | denoise | codec decode |
| --- | --- | --- | --- |
| MLX fp16 | 135 | 132 | 479 |
| Core ML fp16 | 156 | 137 | 231 (T = 250) |

Both match fp32 (argmax agreement ~0.99). Core ML compiles the backbone on
first load (~20 s), cached afterwards; the codec ships precompiled (the
earlier uncompiled one took 69 s). Core ML fp16 `log_softmax`
underflows to -inf for probabilities below 1e-38 (codebooks 0-1): harmless for
argmax, clamp before any guidance arithmetic.

## Reproduce

```
PYTHONPATH=src <fork venv>/bin/python -m exporter.cli export echo --backend xnnpack --output-dir <dir>
uv run --project exporter_models/echo exporter export echo --backend coreml --output-dir <dir>
PYTHONPATH=src <fork venv>/bin/python -m exporter.cli export echo --backend mlx --output-dir <dir>
PYTHONPATH=src <fork venv>/bin/python -m exporter.cli export echo --backend vulkan --output-dir <dir>
```

`multi_pte`: each command writes every file of that backend (`--local-spec`
with a one-variant config exports one). XNNPACK and Vulkan output is
deterministic (the three xnnpack files match the published ones' md5); MLX differs only in a 4-hex prefix of its named-data keys,
and Core ML embeds a random model id, so neither repeats byte for byte.
