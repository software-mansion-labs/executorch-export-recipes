# The four recipes

Each preset reproduces a PTE that was actually built. This file records what
each one is, where it came from, and how far to trust it.

## Provenance

| Preset | Source script | Ships as | Trust |
|---|---|---|---|
| `xnnpack-text` | `iter210_text_xnnpack_export.py` | RN text bundle, `xnnpack/` | validated on device |
| `xnnpack-mm` | `iter211_mm_xnnpack_export.py` | RN multimodal bundle, `xnnpack/` | validated on device |
| `vulkan-text` | `iter212_text_vulkan_export.py` | RN text bundle, `vulkan/` | validated on device |
| `vulkan-mm` | `iter215_mm_vulkan_tds768_export.py` | **not shipping** — see below | **unvalidated** |

The source scripts remain in `gemma_export/experiments_vulkan/op_bisect/`. This
package is a copy, not a move; the archive is untouched.

### What ships today vs what `vulkan-mm` builds

The React-Native multimodal Vulkan bundle is byte-identical to `iter214`, whose
recorded verdict is **text coherent, vision and audio emit whitespace only**.

The cause is a bound, not a numerics bug. `iter214` set the `text_decoder`
sequence bound to 128 while the vision encoder emits 280 soft tokens and the
audio encoder up to 744. `extension/llm/runner/multimodal_prefiller.cpp:217-229`
reads `seq_len` straight from the encoder output and calls `execute()` in one
shot — no chunking, no bounds check — so exceeding the bound yields silent
garbage rather than an error.

`iter215` changes exactly one constant: `TEXT_DECODER_PREFILL_LEN` 128 → 768.
That is the understood fix. It was never run on a device. `vulkan-mm` carries
the 768 value and is flagged `validated_on_device=False`, which raises warning
R13 on every export.

To reproduce exactly what ships today, including its failure, you must pass
`--force` — validation rule R5 refuses this configuration by design, because it
is the bug:

```
python -m gemma4.export --recipe vulkan-mm --text-decoder-prefill-len 128 --force
```

---

## Sequence lengths — the part that is not intuitive

Three different bounds, doing three different jobs:

| Knob | Sizes | Notes |
|---|---|---|
| `max_seq_len` | the **KV cache** | total generation budget. Cheap to raise. |
| `prefill_len` | the `token_embedding` / single-`forward` seq dim | must cover the full multimodal id sequence in the multimodal recipes |
| `text_decoder_prefill_len` | the `text_decoder` seq dim | **this is what the activation arena scales with** |

Holding the prefill bounds low while raising `max_seq_len` buys long generation
almost for free. Raising a prefill bound is what costs arena. Measured at
iter201 (`prefill_len=2048`, `text_decoder_prefill_len=128`):

| Method | Arena |
|---|---|
| text_encoder | 82.02 MB |
| text_decoder | 80.94 MB |
| vision_encoder | 11.35 MB |
| audio_encoder | 6.37 MB |

`text_encoder`'s arena tracked `prefill_len` (20.50 → 41.01 → 82.02 MB across
512 / 1024 / 2048) while `text_decoder` stayed pinned at 80.94 MB, because the
two dims are independent.

### `get_max_seq_len` is not the context limit

The runner reads `get_max_seq_len` as the **per-call prefill chunk size**;
`get_max_context_len` is the context limit. Every recipe here bakes
`get_max_seq_len` = the graph's `text_decoder` seq bound, deliberately, so the
runner chunks at exactly what the graph accepts. The consequence is that a
single prefill call is capped at that value and longer prompts need runner-side
chunking. `validate.py` rule R8 states this on every export.

---

## Quantisation

| | `xnnpack-text` | `xnnpack-mm` | `vulkan-text` | `vulkan-mm` |
|---|---|---|---|---|
| Linear | 8da4w int4 | 8da4w int4, skip `q/k/v/o_proj` | 8da4w int4 | 8da4w int4 |
| Linear group | 128 | 128 | 64 | 128 |
| Main embedding | int4 / g32 | int4 / g32 | int4 / g64 | int4 / g32 |
| PLE embedding | int4 / g32 | int4 / g32 | int2 / g32 | int2 / g32 |
| Base dtype | fp16 | fp16 | bf16 → fp16 retarget | bf16 → fp16 retarget |
| Logits | fp16 | fp16 | **fp32** | fp16 |
| Vision | — | 8da4w | — | **fp32, unquantised** |
| Audio | — | fp32 | — | fp32 |

Notes on the non-obvious entries:

- **QKVO excluded** in `xnnpack-mm`. Bisected 2026-04-23: quantising q/k/v/o_proj
  on top of the MLPs compounded drift enough to flip the sampled token
  (10979 → 9366). They stay fp16.
- **Audio stays fp32.** An fp16 audio PTE measured `min_cos = 0.85` on a row that
  flipped argmax 818 → 100, even though eager fp16 measured 0.99996.
- **Vision stays fp32 on Vulkan** (the iter193b finding). This is roughly 3.5 GB
  of the 4.37 GB PTE and is the single largest remaining size lever. Re-enabling
  it re-introduces the `aten.clamp.Tensor` problem — see `LEVERS.md`.
- **`vulkan-text` leaves logits fp32** while the rest of the model is fp16. That
  is what iter212 does. It is preserved rather than "fixed", because changing it
  is a numerics change nobody has measured.
- **fp16 retarget** (Vulkan only) rewrites torchao's quantised stack to fp16
  while upcasting the scales to fp32. The Vulkan shaders are fp16; the scales
  must stay fp32 or dequantisation loses range.
- **PLE int2** saves roughly 580 MB but was measured to flip the multimodal
  first-token argmax (eager PLE int2 cos 0.895). `xnnpack-text` defaults to int4
  here even though iter210 shipped int2; pass `--ple-bits 2` for iter210 parity.

Eager cosine floors, for gating: text 8da4w vs fp32 is **0.95–0.97**, audio is
**0.92** (not 0.95 — that is the text figure). PLE int2 is 0.895. Gate
PTE-vs-eager-quant at 0.99 and use on-device coherent text as the real signal.

### `audio_k_max` — a discrepancy resolved in favour of 62

The two multimodal source scripts disagree:

| Source | `k_max` |
|---|---|
| iter211 (XNNPACK) | `AUDIO_K_MAX = 63` |
| iter215 (Vulkan) | `AUDIO_SAMPLE_K_MAX = BLOCK_K_MAX = 62` |
| `model/audio.py` | `BLOCK_K_MAX = 62` |

Both recipes here use **62**. iter211's 63 exceeds the model's own
`BLOCK_K_MAX` constant, so its `Dim("audio_blocks", max=63)` admits a `k` the
audio tower's baked buffers were not sized for. That looks like an off-by-one
rather than a deliberate stretch, and 62 (29.76 s at 16 kHz, T_soft = 744) is
what `MODEL_INTERFACE.md` and the runner both assume.

---

## Backends per method

`xnnpack-text`, `xnnpack-mm`: every method uses
`[XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]`, DQ first.
Order matters — without the DQ partitioner first, `XnnpackPartitioner`
const-folds the dequantised weights into its delegate blob and the size saving
disappears entirely.

`vulkan-text`, `vulkan-mm`: one shared

```python
VulkanPartitioner(
    compile_options={"require_dynamic_shapes": True},
    operator_blocklist={exir_ops.edge.torchao.dequantize_affine.default},
)
```

instance across `token_embedding`, `vision_encoder` and `text_decoder`. No
storage-type override, no memory-layout override, no fp16 flag, no allowlist.

`audio_encoder` runs on **XNNPACK even inside the Vulkan PTE**, because the
Vulkan `gemma_custom_sdpa` per-head + softcap shader measured cos 0.40.
`validate.py` rule R10 refuses `audio_backend="vulkan"`.

---

## Methods and signatures

Text recipes export a **single method named `forward`**. The name is not
optional: `extension/llm/runner/text_decoder_runner.h` hardcodes
`load_method("forward")`, and a PTE whose only method is called anything else
aborts at load.

| Method | Inputs | Output |
|---|---|---|
| `forward` | `input_ids` int64 `(1,S)`, `input_pos` int64 `(S,)` | logits |
| `token_embedding` | `input_ids` int64 `(1,S)` | `(embeds, ple_tokens)` |
| `text_decoder` | `embeds`, `ple_tokens`, `input_pos` int64 `(S,)` | logits |
| `vision_encoder` | `pixel_values` fp32 `(1,3,448,640)`, **static** | vision embeds `(1,280,·)` |
| `audio_encoder` | `waveform` fp32 `(1, 7680·k)`, `num_valid_samples` int64 | audio embeds `(1,T_soft,·)` |

Audio waveform length must be **exactly** `7680 · k` for `k ∈ [1,62]`. Vision
input must be contiguous fp32 in the 0–255 range — `method.execute()` on a
non-contiguous view silently reinterprets the buffer against the PTE's expected
strides and yields cos ≈ 0.7 with no error.

---

## Backend prerequisites

The Vulkan recipes depend on patched `backends/vulkan/` kernels — the softmax
max-subtraction fix, the gelu/tanh fp16 clamp, `GemmaSDPA.cpp` implementing the
two `et_vk::` ops, and several `resize_fn` fixes. Those live in a separate
repository and are **not** vendored here.

A PTE exported against one set of shaders and run against another produces
gibberish at runtime, not an error. Export and runtime must come from the same
tree state.
