## What

Adds Vulkan exports for two models, and one cross-backend speedup to the whisper preprocessor that fell out of profiling them.

- `all_minilm_l6_v2` Vulkan fp32
- `whisper` Vulkan fp16 (encode on Vulkan, decode on XNNPACK)
- whisper mel preprocessor: STFT magnitude without a reduction, which helps every backend

## all_minilm_l6_v2

On a Mali-G76 (Galaxy S10+), at the published 254 token shape, interleaved and order-reversed over 4 rounds of 30 executions on a cooled device:

| backend | median | min | p90 | load |
| --- | --- | --- | --- | --- |
| Vulkan | **169.8 ms** | 163.8 | 179.4 | 1685 ms |
| XNNPACK | 408.4 ms | 306.3 | 496.9 | 131 ms |

2.4x on medians. XNNPACK drifts upward across rounds (382 to 444 ms) while Vulkan stays flat, so the conservative read comparing each arm's best case is 1.9x. Output is bit-identical across 10 executions at cosine 0.9999924 against the CPU reference.

The recipe passes `attn_implementation="eager"` rather than using the default SDPA path. SDPA builds its additive mask with `-inf`, and the Vulkan graph serializer writes floats through flatc's JSON, which has no spelling for a non-finite value, so the export dies with `cannot parse value starting with: -`. Eager builds the same mask from finite values, and it also partitions better: the whole model lowers to a single delegate blob.

## whisper

Encode goes to Vulkan, decode stays on XNNPACK. Galaxy S26 Ultra (Adreno 840), whisper-tiny encode, 18 samples per arm interleaved and order-reversed over 6 cooled rounds:

| backend | median | min | p90 |
| --- | --- | --- | --- |
| Vulkan | **61.9 ms** | 56.7 | 68.0 |
| XNNPACK | 149.7 ms | 145.0 | 154.0 |

2.42x. Cosine 0.99994 against the CPU reference, max abs error 1.21 on an output whose own max is 19.3.

Decode is left on XNNPACK deliberately. It is a single-token step, so it is latency-bound and never fills the GPU: 10.7 ms there against 15.8 ms on Vulkan.

Encode is exported at a static 30 s, because Vulkan has no dynamic-shape support. `vulkan/schema.json` carries that narrower contract instead of the 201..480000 range the other backends accept.

Encoder self-attention is re-expressed over rank-3 tensors. Vulkan packs the trailing dimension into texels, so a rank-4 `[1, heads, ctx, dim]` attention spends three quarters of every texel on the singleton batch. Folding the batch into the head dimension is mathematically identical and much faster. It is bound per block instance rather than on the class, because the decoder shares that class and does pass a causal mask.

An XNNPACK partitioner follows the Vulkan one to catch the mel preprocessor. Its STFT is a conv1d over the full 480000-sample waveform, whose width-packed texture would need 120100 texels against a 16384 device limit, so Vulkan cannot take it at any precision.

## STFT magnitude without a reduction

The preprocessor built its magnitude spectrogram as `stft.view(n_freqs, 2, -1).pow(2).sum(dim=1)`, relying on the STFT kernel interleaving each frequency's cos and sin rows. No backend delegates `sum()`, so it ran on the portable CPU kernel and, worse, split the surrounding graph: the tensor had to be copied out of the delegate before it and back in after.

Emitting the two halves as contiguous channel blocks lets them be sliced apart and combined with mul/add instead. Bit-identical output, verified exactly equal on real audio, randn, zeros and a 50x-amplitude signal. Preprocessor alone, 30 samples per arm over 6 cooled rounds:

| arm | before | after |
| --- | --- | --- |
| Vulkan + XNNPACK | 22.52 ms | **9.04 ms** |
| XNNPACK only | 69.32 ms | **60.64 ms** |

That is where most of whisper's encode win comes from on both backends: Vulkan 72.97 to 61.91 ms, XNNPACK 160.22 to 149.67 ms.

The other portable-CPU op left in the preprocessor is the `max()` feeding the dynamic-range clamp. Measured at 1.09 ms, so it is left alone.

This changes no published .pte on its own. The other whisper backends pick the win up whenever they are next re-exported, with identical output.

## Runtime dependency

Both exports pin `executorch` to the labs fork branch `@ms/separate-backends-1.4.1`, which carries Vulkan work also filed upstream. Each `vulkan/pyproject.toml` lists what it is waiting on, and the pins can drop back to a released executorch once those land.

whisper needs, as AOT:

- conv1d lowered as conv2d over a singleton height (pytorch/executorch#22330). Without it the encoder's conv stem takes Vulkan's conv1d kernel, which runs at ~3% of the GPU's matmul rate (#22329) and leaves the whole encode slower than XNNPACK.

and as runtime:

- broadcast size calculation for 0-dim tensors (#22332), without which encode aborts with `std::out_of_range`
- local work group shape on Adreno (#22328), without which the encoder is nondeterministic run to run (#22327)

all_minilm_l6_v2 needs no fork-only AOT pass, but its runtime needs the reduce shader barrier fix (#22326), the int32 to uint8 `view_convert` variant (#22321), and the `ne`/`lt`/`le`/`gt`/`ge` scalar op registrations.

## Published

`vulkan/all_minilm_l6_v2_vulkan_fp32.pte` is on the HF repo at `main` and the `v0.10.0` tag, wired into the registry in react-native-executorch#1392. The whisper Vulkan file is not published yet.

## Noticed, not fixed here

`xnnpack/tiny/config.json` declares whisper decode as `[1, 128]` tokens while the export builds it static at `[1, 1]`, which is also what the .pte's embedded schema says. The new `vulkan/tiny/config.json` declares `[1, 1]`. The xnnpack one looks wrong but predates this branch.

## qwen3 0.6B: exporter added, nothing published

`qwen3/vulkan/export.py` produces a working, verified 0.6B 8da4w artifact and no `.pte` is published. It wins one axis decisively on one GPU and loses the others, so there is no single artifact worth shipping yet. Full numbers and the recipe are in `exporter_models/qwen3/NOTES.md`.

Galaxy S26 Ultra (Adreno 840), 206-token prompt, against the published XNNPACK 8da4w:

| backend | prefill tok/s | decode tok/s | RSS |
| --- | --- | --- | --- |
| Vulkan | **1685** | 71 | **303 MiB** |
| XNNPACK | 627 | **82** | 1088 MiB |

2.7x prefill, 0.86x decode, time to first token 93 ms against 344 ms.

Galaxy S10+ (Mali-G76) inverts it: prefill 11.6 against 52.6, decode 12.0 against 7.2. The decode win grows to 3.4x at a 773-token prompt because Vulkan decode is nearly flat with context while XNNPACK collapses. Prefill costs about as much per token as decode on Mali, so batching the prompt buys almost nothing there, against a 24x batching benefit on Adreno.

Three settings carry the result: a plain partitioner that lets the `causal_sdpa` pattern fuse attention back in, which collapses 57 delegates into one; `enable_dynamic_shape`, without which `TextPrefiller` prefills one token at a time; and `force_fp16`, which turns out to be an Adreno-only win and a Mali loss on both axes.

Two ExecuTorch fixes fell out of it. pytorch/executorch#22477 (PR #22479) is required to reproduce the export at all: the memory planner asserts `graph_output_allocated not set` on a graph whose output carries no tensors, which every dynamic-shape Vulkan LLM produces. pytorch/executorch#22478 (PR #22480) fixes a Vulkan delegate that declares fewer inputs than the call passes.
