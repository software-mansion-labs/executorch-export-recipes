# qwen3_asr notes

Qwen3-ASR-0.6B speech recognition ([react-native-executorch#769](https://github.com/software-mansion/react-native-executorch/issues/769)):
an 18-layer audio tower in front of a Qwen3-0.6B decoder. The model code is a
local port in `_qwen3_asr/` (audio tower + mel front end written here, decoder
built on the shared llama `construct_transformer`), loaded straight from the
`Qwen/Qwen3-ASR-0.6B` safetensors. Upstream ships no `.pte`, so every
`{backend}/{size}/config.json` is local.

## The .pte contract

Four methods plus constants, and a baked `get_model_schema` (`schema.json`):

| method | in | out |
|---|---|---|
| `preprocess` | `f32[T]` 16 kHz, T 401..480000 | `f32[1,128,T//160]` log-mel |
| `audio_encoder` | `f32[1,128,100c]`, `i64[1]` n_valid, c 1..8 | `f32[1,13c,1024]` |
| `token_embedding` | `i64[1,S]`, S 1..1024 | `f32[1,S,1024]` |
| `text_decoder` | `f32[1,S,1024]`, `i64[1]` start_pos, S 1..1023 | `f32[1,151936]` last-token logits |

The runtime builds the prompt (`PROMPT_PREFIX` + audio embeddings +
`PROMPT_SUFFIX`, optionally followed by `language <Name><asr_text>` to force a
language), prefills, and decodes greedily. The model answers
`language English<asr_text>...`; `language None<asr_text>` means no speech.
`get_max_context_len` (1024) caps one pass at 30 s of audio.

## Port decisions

- **Mel front end** reproduces `WhisperFeatureExtractor` at 128 bins (power
  spectrum, slaney mel, log10, global max-8 floor, last frame dropped). STFT is
  a conv1d and the reflect pad is a matmul with an anti-identity, so it
  delegates. Matches the reference to 3e-5.
- **Chunking.** The tower convolves every 100-frame (1 s) chunk independently,
  13 tokens per chunk; the tail chunk is zero-padded in mel space and only its
  real tokens are kept (`audio_token_count`).
- **Windowed attention.** Tower attention is limited to 8 s windows (800 frames,
  104 tokens), as in the FA2/vLLM paths. transformers' sdpa path attends
  globally instead; the two agree up to 8 s. `audio_encoder` takes one window
  per call, and `n_valid` masks the padded tail.
- **Decoder** maps onto the llama transformer (qk-norm, rope theta 1e6). The
  MRoPE positions are equal across sections for audio-only input, so plain 1-D
  RoPE is exact.

fp32 parity with the reference: encoder 2e-7, identical transcripts.

## Quantization: decoder layer 0 must stay at 8 bits

4-bit weights on decoder layer 0 make the first generated token unreliable.
On some clips the model then skips its `language <Name><asr_text>` header
and derails ("uttermost", repeated Chinese). The fp32 model puts that token at
log p ≈ 0 on every clip, with a 7+ nat margin. 8da4w-g32 on every decoder
linear drops it to −1.8 on the worst of 8 failing clips. Keeping layers 0–1 at
8 bits brings it back to −0.05. Later layers and the output head matter far
less once layer 0 is fixed, and group 64 loses more than group 32. Per backend:

| backend | decoder layers 0–1 | rest of decoder and head |
|---|---|---|
| xnnpack | 8da8w per-channel | 8da4w g32 |
| vulkan | fp (fp16 on GPU) | 8da4w g32 |
| mlx | 8w per-channel | 4w g32 |

XNNPACK `.pte` WER on the first 100 LibriSpeech test-clean clips (greedy, the
RNE pipeline in Python, `tools/pte_run.py` + `tools/wer.py`):

| recipe | WER | derailed clips |
|---|---|---|
| fp32 | 2.14% | 0 |
| decoder all 8da4w-g32 | 7.22% | 4 |
| decoder all 8da4w-g32, int8 head | 8.24% | 4 (different ones) |
| decoder layers 0–1 and head int8 | 2.41% | 0 |
| decoder layers 0–1 int8 | 2.35% | 0 |
| + int8 conv stem (shipped) | 2.46% | 0 |
| + tower linears 8da4w-g32 | 2.83% | 0 |

The 4-bit head is also 65 MB smaller and ~4% faster per decoded token on a Mac.
The tower's linears stay at 8da8w: 4 bits costs ~0.5 WER points.

Vulkan has no dynamically quantized 8-bit per-channel linear, so those layers
stay in the graph dtype. `tools/quant_sweep.py` reproduces the sweep without an
export.

## xnnpack

`_qwen3_asr/transforms.py` rewrites ops XNNPACK cannot take into ones it can:
LayerNorm/RMSNorm means become a linear with a constant `1/D` row, and RoPE's
`rotate_half` becomes a linear with a signed permutation. Tower attention uses
`bmm` on `[heads, N, head_dim]` instead of SDPA, so there is no masked-row guard
or `expand_copy` on the host. The windowed mask is computed from a constant
index buffer with `clamp`, with no `arange`/`where`. All rewrites are exact
(encoder Δ1.6e-7, logits Δ5e-5).

The conv stem is about half the tower's FLOPs (~34 GFLOP per 8 s window) and 65%
of its fp32 time. `conv:8da8w` runs it as XNNPACK dynamically quantized int8
conv2d (PT2E, per-channel weights): tower output cos 0.99994, tower 1.33x faster
on the S26 Ultra.

The decoder uses the llama custom KV cache and `custom_sdpa`, which run as CPU
kernels by design. `RemoveInversePermutePairs` (`_qwen3_asr/passes.py`) drops the
cache transpose pair around them: without it two full-cache transposes per layer
were ~40% of a decode step (S26 Ultra: 44 -> 20 ms/token in a sustained run,
identical transcripts). What stays on the host besides them is views,
`select_copy`, and scalar reads.

## vulkan: exports, not shipped

The decoder lowers to one Vulkan delegate once `RemoveInversePermutePairs` lets
`causal_sdpa` fuse (same pass as lfm2_5_vl). The tower's three conv2d layers
take the chunks as their batch, but Vulkan convolution takes batch 1 only, so a
second `XnnpackPartitioner` picks them up. The 4-bit embedding needs
`packed=True` to lower to `embedding_q4gsw`. Needs the software-mansion-labs
fork (`vulkan/pyproject.toml`).

On the S26 Ultra it loses to XNNPACK everywhere but prefill (tables below).
ET-VK's fp16 linear reaches ~600 GFLOP/s here (`linear_vec_bias_half` is 80% of
the tower's GPU time), below what this CPU does in int8. Moving the conv stem to
the GPU (batched conv2d) would be slower than the CPU for the same reason. The
lever is ET-VK GEMM throughput on Adreno, not the partitioning. Decode on the GPU
is 15.2 ms per step: 4-bit linears 45%, the fp16 layers 0–1 16%, per-row
activation quantization 7%, RMSNorm 7%, layout copies 6%.

`quantize` takes `enc:8da4w` (tower linears, needs pytorch/executorch#24055) and
`conv:8da8w`. XNNPACKQuantizer cannot target convs by module type until
pytorch/executorch#24056, hence the name filter in `exportable.py`.

## mlx

`attention_type="mlx"` builds MLX's own attention and KV cache. Compute is
bf16 (fp16 for the tower) behind fp32 method IO (`_CastIO`), so the runtime
exchanges fp32 tensors with every backend. Needs the software-mansion-labs fork
(`mlx/pyproject.toml`). Not measured on device yet.

## Device: Galaxy S26 Ultra (Adreno 840)

`tools/bench` is the RNE pipeline in C++ (build notes in its CMakeLists.txt);
`tools/device_wer.py` turns its log into transcripts and medians.

Cool device, 8 s bursts with 30 s cool-downs, alternating order, 24 clips (3-17 s),
medians, before the int8 conv stem:

| backend | encoder | prefill | decode | whole clip |
|---|---|---|---|---|
| xnnpack | 71.8 ms | 88.7 ms | 12.5 ms/token | 360 ms |
| vulkan | 109.5 ms | 63.6 ms | 15.4 ms/token | 451 ms |

Tower only, cool device, medians of 6:

| variant | 8 s window | 3.5 s clip |
|---|---|---|
| xnnpack | 66.2 ms | 37.9 ms |
| xnnpack, int8 conv stem (shipped) | 49.7 ms | 28.7 ms |
| vulkan fp16 | 100.4 ms | 62.0 ms |
| vulkan, int8 conv stem on CPU | 91.1 ms | 54.9 ms |
| vulkan, int8 weight-only linears | 87.2 ms | 50.5 ms |
| vulkan, 8da4w linears | 69.5 ms | 43.7 ms |

All 100 clips back to back, two rounds in alternating order (sustained load; the
GPU throttles far harder than the CPU, so Vulkan decode drifts from 15 to 35 ms):

| backend | WER | encoder | prefill | decode |
|---|---|---|---|---|
| xnnpack (before the transpose fix) | 2.46% | 127-130 ms | 234-244 ms | 43.7-45.3 ms/token |
| xnnpack | 2.46% | 115-128 ms | 184-208 ms | 19.4-20.1 ms/token |
| vulkan | 2.51% | 196-215 ms | 132-151 ms | 32.9-38.4 ms/token |
