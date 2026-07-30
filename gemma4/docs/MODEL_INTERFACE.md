# PTE runtime contract

What a runner must do to drive a Gemma4 PTE produced by this package. Written
for someone implementing or debugging a runner, not for someone exporting.

Refreshed against iter213–215. Where the older `gemma_export/MODEL_INTERFACE.md`
(2026-05-18, written for iter201) disagrees, this file is correct — the
differences are listed at the bottom.

---

## Methods

### Text recipes — single method

```
forward(input_ids: int64[1, S], input_pos: int64[S]) -> logits[1, S|1, vocab]
```

The method **must** be named `forward`: `extension/llm/runner/text_decoder_runner.h`
hardcodes `load_method("forward")`, and a PTE whose only method has another name
aborts at load.

`S` is dynamic in `[1, prefill_len]`. `min=1` so one method serves both prefill
and single-token decode.

### Multimodal recipes — four methods

| Method | Inputs | Outputs |
|---|---|---|
| `token_embedding` | `input_ids` int64 `[1, S]`, S ∈ [1, prefill_len] | `embeds [1,S,hidden]`, `ple_tokens [1,S,num_layers,ple_dim]` |
| `vision_encoder` | `pixel_values` fp32 `[1,3,448,640]` — **static** | vision embeds `[1,280,hidden]` |
| `audio_encoder` | `waveform` fp32 `[1, 7680·k]`, `num_valid_samples` int64 | audio embeds `[1, T_soft, hidden]` |
| `text_decoder` | `embeds`, `ple_tokens`, `input_pos` int64 `[S]`, S ∈ [1, text_decoder_prefill_len] | `logits [1,·,vocab]` |

Output dtypes vary by recipe. Read them at runtime; do not assume. On the Vulkan
path `embeds` and the audio encoder output are `Half`; on fp32 paths they are
`float`.

---

## Calling order

1. Load the module and its methods.
2. If audio present: validate the byte count, run `audio_encoder`, keep the
   output EValue alive.
3. If image present: run `vision_encoder`, keep the output alive.
4. Build `input_ids` per the chat template below, recording where each modality's
   soft-token span starts.
5. Run `token_embedding` → `(embeds, ple_tokens)`.
6. Splice the encoder outputs into `embeds` at the recorded offsets.
7. Prefill `text_decoder` in chunks of ≤ the graph's seq bound.
8. Sample from `logits[:, -1, :]`, then decode autoregressively, one token at a
   time, incrementing `input_pos`.

### The splice

At every position that will receive a vision or audio splice, `input_ids` must
contain **`0`** (the pad token), *not* the placeholder id. The placeholder ids
(258880 / 258881) are for tokenisation only. `ple_tokens` is **not** spliced —
it stays exactly as `token_embedding` produced it, which correctly used
`pad_token_id=0` for its lookup.

```cpp
// dst = embeds [1, S, hidden]; src = vision or audio embeds [1, count, hidden]
for (i = 0; i < count; ++i)
    memcpy(dst + (start + i) * hidden, src + i * hidden, hidden * sizeof(T));
```

Handle the asymmetric `Half(dst) ← Float(src)` case with a per-element cast.
Audio and image spans touch disjoint positions, so splice order does not matter;
convention is prompt order (audio, then image).

---

## Chat template

| Phase | Tokens |
|---|---|
| prefix | `<bos><\|turn>user\n` = `[2, 105, 2364, 107]` |
| audio span | `[256000, 258881 × T_soft, 258883]` |
| image span | `[255999, 258880 × 280, 258882]` |
| text | tokenised prompt |
| generation prompt | `<turn\|>\n<\|turn>model\n` = `[106, 107, 105, 4368, 107]` |

No `\n\n` padding around spans. When both modalities are present, EOA is
followed **immediately** by BOI with no separator: `[258883, 255999]`.

### Token ids

| Token | Id |
|---|---|
| `<bos>` | 2 |
| `<\|turn>` / `<turn\|>` | 105 / 106 |
| `\n` | 107 |
| `<\|image>` / `<\|image\|>` / `<image\|>` | 255999 / 258880 / 258882 |
| `<\|audio>` / `<\|audio\|>` / `<audio\|>` | 256000 / 258881 / 258883 |
| `user` / `model` | 2364 / 4368 |
| `pad_token_id` | 0 |

These are the **local Gemma4 fork's** ids. Upstream `gemma3n` uses different
values (audio 262273, image 262145) — do not use those. The tokenizer must come
from the checkpoint directory, not from a public checkpoint.

---

## Audio input

Raw fp32 little-endian mono PCM at 16 kHz. Byte count must be exactly
`7680 · k · 4` for `k ∈ [1, 62]` — any other length is rejected. Round up to the
next multiple of 30,720 bytes (0.48 s) and zero-pad.

```
ffmpeg -i input.mp3 -ac 1 -ar 16000 -f f32le -y output.f32
```

`T_soft` as a function of sample count:

```
padded  = N + 160                      # semicausal pre-pad
frames  = (padded - 321) // 160 + 1
t = frames
for _ in range(2):                     # two stride-2 conv layers
    t = (t + 2 - 3) // 2 + 1
T_soft  = min(t, 750)
```

| k | samples | duration | T_soft |
|---|---|---|---|
| 1 | 7,680 | 0.48 s | 12 |
| 8 | 61,440 | 3.84 s | 96 |
| 26 | 199,680 | 12.5 s | 312 |
| 52 | 399,360 | 24.96 s | 624 |
| 62 | 476,160 | 29.76 s | 744 |

---

## constant_methods

| Key | Meaning |
|---|---|
| `get_max_seq_len` | **per-call prefill chunk size**, equal to the graph's `text_decoder` seq bound. Not the context limit. |
| `get_max_context_len` | KV cache depth — the actual context limit |
| `get_bos_id` / `get_eos_ids` | `2` / `[1, 106, 50]` (a list) |
| `get_vocab_size` | 262144 |
| `use_kv_cache` / `enable_dynamic_shape` | both true |
| `has_ple` | true on multimodal |
| `image_placeholder_id` / `audio_placeholder_id` | 258880 / 258881 — for tokenisation, **not** for `input_ids` substitution |
| `audio_block_samples` | 7680 |
| `audio_block_k_min` / `audio_block_k_max` | 1 / 62 |
| `audio_sample_count_min` / `_max` | 7,680 / 476,160 |
| `audio_soft_tokens_max` | 744 |

---

## Footguns

| Item | Rule |
|---|---|
| Multimodal slot ids | use `0`, never the placeholder id |
| Prefill chunk size | ≤ the graph's `text_decoder` seq bound |
| Bounds checking | `multimodal_prefiller.cpp` does **none** — exceeding the bound gives silent garbage, not an error |
| Audio byte count | exactly `7680 · k · 4`, `k ∈ [1,62]` |
| Image input | NCHW fp32 `[1,3,448,640]`, raw 0–255 range |
| Input contiguity | `execute()` on a non-contiguous view reinterprets the raw buffer against the PTE's expected strides. Observed cos ≈ 0.7, no error. Always `.contiguous()`. |
| Tokenizer | the local fork from the checkpoint dir, not upstream `gemma3n` |
| Splice destination | mutate `embeds.mutable_data_ptr()`; method-owned memory persists until the next `execute` on that method |
| `input_pos` | absolute KV positions, monotonic across prefill and decode |
| Backend match | PTE and runtime must come from the same kernel tree; mismatch gives gibberish, not an error |

---

## Corrections vs the 2026-05-18 archive copy

The older `gemma_export/MODEL_INTERFACE.md` was written for iter201. Since then:

| Item | Old | Current |
|---|---|---|
| audio embeds dtype | fp32 "regardless of text_encoder dtype" | `Half` on the Vulkan path |
| soft-token key | `audio_soft_tokens` | `audio_soft_tokens_max` |
| `audio_blocks_max` | 63 | 62 |
| block-samples key | `audio_samples_per_block` | `audio_block_samples` |
| dropped keys | — | `audio_block_frames`, `audio_sample_rate` were dropped by iter214; this package re-adds `audio_sample_rate` |
| first method name | `text_encoder` | `token_embedding` (renamed at iter213) |
| `libvulkan_backend.a` md5 | recorded a specific hash | stale; kernels now live in a separate repo |
