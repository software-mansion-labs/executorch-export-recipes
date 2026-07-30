# Optimisation levers — confirmed, falsified, and unclaimed

A record of what was tried on the Gemma4 export, what it measured, and what
therefore should not be tried again. The falsified entries matter more than the
confirmed ones: a lever that sounds obviously good and is actually a regression
will otherwise be rediscovered every few months.

All numbers below are **measured**, not projected, unless explicitly labelled.
Where a number came from a 5-layer truncated PTE it says so — 5L numbers do not
scale linearly to 35L.

---

## Confirmed and applied

| Lever | Flag | Measured effect | Applied in |
|---|---|---|---|
| Broadcast GQA | `use_broadcast_gqa` | prefill −62 to −77% | both XNNPACK recipes |
| Custom SDPA (llama) | `use_custom_sdpa` | decode −29% | both XNNPACK recipes |
| A′ — RMSNorm rsqrt | `RMSNorm.use_rsqrt` | ~−2% prefill and decode each | all four recipes |
| B+C — shared ring state | `use_shared_ring_state` | part of the −671 op total below | XNNPACK recipes |
| E — skip redundant RoPE slice | `skip_redundant_rope_slice` | −20 host ops @5L | XNNPACK recipes |
| Combined A′+B+C+E | — | **text_decoder host ops 2676 → 2005 (−671, −25.1%); arena 464.35 → 382.14 MB (−82.21, −17.7%); top-1 = 9259 "Hello"** | `xnnpack-mm` (via exp115 → iter211) |
| PLE split for export | `split_ple_for_export()` | prerequisite for quantising PLE separately | all four |
| `seq_dim.max` decoupled from `max_seq_len` | `prefill_len` / `text_decoder_prefill_len` | text_decoder arena 12.85 GB → 856 MB (~15×) | all four |
| Vision B1 — split-param position embedding | `use_embedding_position_emb` | part of arena 91 → 66 MB | multimodal recipes |
| Vision B3 — persistent static mask | `use_static_mask` | part of arena 91 → 66 MB | multimodal recipes |
| Vision B5 — fused `et_vk.gemma_custom_sdpa` | `use_vulkan_sdpa` | delegates 36 → 4, arena 66 → 10 MB | `vulkan-mm` |
| Audio rsqrt | audio `RMSNorm.use_rsqrt` | host ops −218 (−13.3%), delegates −109 (−19.1%) | audio path |
| Audio softplus pre-bake | `bake_softplus_scale` | host ops 272 → 212 (−60, −22%) | audio path |
| Audio 5D mask bake | `use_baked_5d_mask` | host ops −22 (−1.5%) | audio path |

---

## Falsified — do not re-apply

### Lever G — `_rotate_half` as a matmul (exp119)

**Removed from this package entirely.** `use_rotate_half_matmul` and
`make_rotate_half_basis` do not exist in `gemma4/model/text.py`.

The idea: `_rotate_half(x)` does `cat((-x2, x1), dim=-1)`, which was attributed
to 3 host ops (`split_with_sizes_copy` + `neg` + `cat`), called twice per layer.
Replacing it with `x @ M` against a precomputed `[D, D]` permutation-and-sign
basis should collapse ~30 host ops at 5L and ~210 at 35L.

Measured at 5L:

| Metric | Before | After | Δ |
|---|---|---|---|
| host ops | 411 | **422** | **+11** |
| delegates | 156 | 171 | +15 |
| seams | 242 | 272 | +30 |
| single-op seams | 193 | 217 | +24 |
| arena | 90.98 MB | 94.27 MB | +3.29 MB |

Eager was bit-identical (`max_abs = 0`, 7/7 top-1 match) — the rewrite is
mathematically exact. It still lost.

Why: the hypothesis assumed `neg` and `cat` were host-side. They were already
being absorbed into XNNPACK delegates as of exp116. Only the 10
`split_with_sizes_copy` ops were genuinely host-side. The matmul removed those
10 but introduced 20 `et_view` reshapes as mandatory 4D→3D→4D boundary glue at
each XNNPACK matmul, netting +11. Projected at 35L: **~+77 host ops.**

Secondary finding worth keeping: caching the basis in a module-level Python
dict caused `torch.export` to capture it under a mangled `_constants` key and
crash `_unsafe_adjust_original_program` with a `KeyError`. Registering it as a
non-persistent `nn.Module` buffer fixed that. If you ever need a derived
constant tensor inside an exported module, register it as a buffer.

### Lever H — dtype-copy deduplication

Ruled out by attribution probe: duplicate dtype casts accounted for only ~1.6%
of ops. Not worth the complexity.

### 4D-reshape fused norm (exp82)

Reverted: **+8 to +16%** regression.

### `F.scaled_dot_product_attention` on Vulkan

Not a lever so much as a trap. It decomposes into a fp32 arena bomb: arena
**91 → 184 MB**, 32× `where.self` on fp32 `(1,12,1120,1120)` plus 32×
`expand_copy` per layer, and fp16 inputs get upcast to fp32 inside the
decomposition. The partitioner never refuses it. Use an explicit `q @ k.T` form
or `et_vk.gemma_custom_sdpa` instead. `Gemma4EdgeVisionConfig.use_fused_sdpa`
still exists and still carries this warning in its own source comment.

### int4 weight-only quantisation on XNNPACK

Measured **1.00×** size reduction — i.e. none. `XnnpackPartitioner`
const-folds the dequantised weights back into its delegate blob. 8da4w with
`XnnpackDynamicallyQuantizedPartitioner` first is the only configuration that
actually shrinks the file.

### Vulkan audio SDPA

The `gemma_custom_sdpa` per-head + softcap shader measured **cos 0.40** against
eager. Audio has run on XNNPACK inside the Vulkan PTE ever since (iter202+).
`validate.py` rule R10 refuses `audio_backend="vulkan"` for this reason.
iter186–191 are the abandoned branch.

---

## Confirmed but never applied — unclaimed wins

### Lever F — pre-broadcast cos/sin (`use_prebroadcast_rope`)

Measured **−32 host ops at 5L** in exp116. Confirmed profitable. It was never
integrated into a production export purely because of ordering: exp115 (the
script that became `xnnpack-mm`) was written at 20:05 and exp116 at 20:24 the
same evening, so exp115 predates the lever and nothing after it re-based.

The flag exists on `Gemma4Attention` and is wired through `AttnFlags`. To claim
it:

    python -m gemma4.export --recipe xnnpack-mm --use-prebroadcast-rope

It has **not** been validated at 35L or with vision/audio in the graph. Validate
before shipping.

### RoPE gather in `decode_from_embeds` — a fix applied in only one of two places

`model/text.py` contains the same Vulkan fix in one method and not the other:

```python
# Gemma4EdgeModel.forward, ~line 1085 — FIXED
# F.embedding lowers to aten.embedding.default (Vulkan-supported);
# `tensor[input_pos]` lowers to aten.index.Tensor which Vulkan rejects.
cos_local = F.embedding(input_pos, self._cos_local)

# Gemma4EdgeModel.decode_from_embeds, ~line 1263 — NOT FIXED
cos_local = self._cos_local[input_pos]
```

`decode_from_embeds` is what `TextDecoderExport` calls, so this is the
`text_decoder` method of **`vulkan-mm`** — four `aten.index.Tensor` nodes per
forward that the Vulkan partitioner declines, forcing a host fallback and the
partition seams that come with it. The single-method `vulkan-text` path goes
through `forward` and is unaffected.

`F.embedding(input_pos, W)` and `W[input_pos]` are numerically identical for a
1-D index, so mirroring the fix should be bit-exact. It is **not applied here** —
this package is an extraction, and changing model behaviour beyond the agreed
Lever G removal is out of scope. Worth doing as a follow-up, with a partition
count taken before and after to confirm the seams actually disappear.

---

## Patches no longer needed

### Scalar-clamp patch (originates exp93, not exp94)

`Gemma4ClippableLinear`'s `input_min`/`input_max` buffers surface as
`aten.clamp.Tensor`, which the Vulkan op registry does not implement (it has
only `aten.clamp.default`), producing a `KeyError` in `insert_prepack_nodes`.
The patch rebound `forward` to use Python floats.

**Not carried into this package.** The edge vision tower's `ClippableLinear`
already uses Python-float bounds via `set_bounds`, and HF
`Gemma4MultimodalEmbedder` contains no `Gemma4ClippableLinear` at all. It
becomes necessary again only if you revert to the HF vision tower or re-enable
vision quantisation — `validate.py` rule R9 warns about the latter.
