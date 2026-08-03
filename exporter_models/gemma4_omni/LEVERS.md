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
| Combined A′+B+C+E | — | **text_decoder host ops 2676 → 2005 (−671, −25.1%); arena 464.35 → 382.14 MB (−82.21, −17.7%); top-1 = 9259 "Hello"** | `xnnpack-mm` |
| PLE split for export | `split_ple_for_export()` | prerequisite for quantising PLE separately | all four |
| `seq_dim.max` decoupled from `max_seq_len` | `prefill_len` / `text_decoder_prefill_len` | text_decoder arena 12.85 GB → 856 MB (~15×) | all four |
| Vision B1 — split-param position embedding | `use_embedding_position_emb` | part of arena 91 → 66 MB | multimodal recipes |
| Vision B3 — persistent static mask | `use_static_mask` | part of arena 91 → 66 MB | multimodal recipes |
| Vision B5 — fused `et_vk.gemma_custom_sdpa` | `use_vulkan_sdpa` | delegates 36 → 4, arena 66 → 10 MB | `vulkan-mm` |
| Audio rsqrt | audio `RMSNorm.use_rsqrt` | host ops −218 (−13.3%), delegates −109 (−19.1%) | audio path |
| Audio softplus pre-bake | `bake_softplus_scale` | host ops 272 → 212 (−60, −22%) | audio path |
| Audio 5D mask bake | `use_baked_5d_mask` | host ops −22 (−1.5%) | audio path |

---

## Confirmed but never applied — unclaimed wins

### Lever F — pre-broadcast cos/sin (`use_prebroadcast_rope`)

Measured **−32 host ops at 5L**. Confirmed profitable.

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
1-D index, so mirroring the fix should be bit-exact. It is **not applied here**.
Worth doing as a follow-up, with a partition count taken before and after to
confirm the seams actually disappear.
