# lfm2_5_vl notes

LFM2.5-VL vision-language model (vision_encoder + token_embedding + text_decoder),
sized `[450m, 1_6b]`, exported via the vendored `_lfm2_5_vl/` package (model code
from the executorch `examples/models/lfm2_5_vl` recipe; shared llama
transformer/kv-cache/sdpa infra kept as an import). One universal `export.py` per
backend reads `config["size"]` and maps it to `LiquidAI/LFM2.5-VL-{450M,1.6B}`.

## Why local config.json (not an HF pointer)

The published `react-native-executorch-lfm-2.5` repo does not ship a
`{size}/{backend}/config.json` for the VL sizes, so the multi-method VL contract
is hand-authored locally per `{size}/{backend}/`. 1.6B shapes scale with the text
hidden dim (2048 vs 450M's 1024): `token_embedding` out `[1,2048,dim]`,
`text_decoder` in `[1,2048,dim]`, `vision_encoder` out `[1,256,dim]`. The 1.6B
`vision_encoder` output dim is derived from the canonical config, not verified
against a shipped `.pte` — the harness will surface any mismatch at real export.

## Export specifics

The vendored `_lfm2_5_vl` already supports both sizes: `_resolve_params_path`
picks the bundled `config/lfm2_5_vl_{450m,1_6b}_config.json` from the model_dir
string. export.py passes `params_path` explicitly because upstream `export_all`
only resolves params in its CLI `main()`.

Producing the `.pte` needs an executorch source build: the software-mansion-labs
fork for `mlx/` and `vulkan/` (both are version-locked to their runtime), a stock
build for `xnnpack/`.

## vulkan — the Android GPU export (#1360)

Added for [react-native-executorch#1360](https://github.com/software-mansion/react-native-executorch/issues/1360),
which asks whether LFM2.5-VL's hybrid architecture lowers to Vulkan at all.
It does, completely:

| method | delegate | host ops |
|---|---|---|
| `text_decoder` | 1 Vulkan call | none |
| `token_embedding` | 1 Vulkan call | none |
| `vision_encoder` | 111 XNNPACK calls (1.6B) | 484 |

The decoder is the answer to the question. Both the `full_attention` layers and
the 10 gated short-conv layers land inside a **single** Vulkan delegate blob —
no fragmentation, no fallback: `aten.convolution` (grouped 1-D), the conv-state
`cat`/`slice_copy`, `aten.rms_norm`, and `et_vk.apply_rotary_emb_hf` are all
claimed by the partitioner, and the attention layers fuse into one
`llama::sdpa_with_kv_cache` node each.

Lowering completely is necessary but not sufficient — see "Two backend bugs this
export uncovered" below, which is what stood between a clean partition and a
model that actually runs.

### The one thing that has to be patched in the export itself

`RemoveInversePermutePairs`, in `_lfm2_5_vl/export_lfm2_5_vl_vulkan.py`. Without
it the export dies inside Vulkan's `FusePatternsPass` on a bare `AssertionError`
that names neither attention nor the KV cache.

`CustomKVCache.update` returns `k_cache.transpose(1, 2)` and `SDPACustom.forward`
transposes straight back; the pair is an identity but both survive into the edge
graph. Vulkan's `causal_sdpa` matcher reads `custom_sdpa.args[1]` and searches
*its* users for the `llama::update_cache` node, so a permute in between makes the
search come up empty. The pass collapses inverse permute pairs at edge level,
before partitioning, which is the graph shape the matcher was written against.
It removed 30 pairs on 450M. This is not LFM2.5-specific — any model taking
llama's `replace_kv_cache_with_custom_kv_cache` + `replace_sdpa_with_custom_op`
path to Vulkan should hit it.

### Recipe, and where it deviates from the XNNPACK sibling

| | xnnpack | vulkan |
|---|---|---|
| decoder linears | 8da4w, group 128 | 8da4w, group **64** |
| token embedding | int8, group 32 | int4, group **64** |
| vision tower | fp32, XNNPACK | fp32, XNNPACK (unchanged) |
| decoder seq bound | 2048 | **256** |
| `get_max_seq_len` | 2048 | 256 |
| `enable_dynamic_shape` | false | **true** |

- **group 64 / int4 embeddings** are not preferences. Vulkan fuses
  `linear_dq8ca_q4gsw` and `embedding_q4gsw` — the 4-bit groupwise shaders —
  and 64 is the group size the device-validated gemma4/vulkan recipe uses. int8
  embeddings emit `embedding_byte`, which no Vulkan pattern claims, so that
  method would fall back to host kernels entirely.
- **seq bound 256 + `enable_dynamic_shape: true`** is the contract
  `multimodal_prefiller.cpp` reads: `get_max_seq_len` is the decoder chunk size
  and the whole prefill is bounded against `get_max_context_len` (2048). 256
  matches the vision encoder's soft-token count, so one image rides in one
  chunk, and keeps activations — which Vulkan holds in GPU memory — an eighth of
  the 2048-wide XNNPACK graph's.

### Quantisation cost, measured

Eager decoder logits versus the fp32 baseline: same 17-token prompt, no image,
each recipe applied as its own export applies it (`get_quant_weight_transform`
for the linears, `EmbeddingQuantHandler` for the table), then cosine over the
logits. Recipe-level, so the numbers are independent of which backend later runs
the graph. No pt2e quantizer is involved on either path — both exports resolve
`get_quantizer_and_quant_params` to an empty list — so this covers the whole
quantisation difference between them:

| | 450M | 1.6B |
|---|---|---|
| vulkan (8da4w g64 + int4/g64) | cos 0.99976, argmax match | cos 0.99787, argmax match |
| xnnpack (8da4w g128 + int8/g32) | cos 0.99947, argmax match | cos 0.99525, **argmax flipped** |

The Vulkan recipe is the *less* lossy of the two at both sizes. On 1.6B the
shipped XNNPACK recipe flips the sampled token on this prompt (509 → 7, which is
`<|im_end|>`) while the Vulkan one holds. One prompt at one position is not a
quality verdict, but it does retire the worry that int4 embeddings would be the
regression — they are not.

### Artifact sizes

| | 450M | 1.6B |
|---|---|---|
| vulkan 8da4w | 656 MB | 2.53 GB |
| published xnnpack 8da4w | 649 MB | 2.43 GB |

Parity. The remaining delta is the fp32 vision tower, which is the single
largest lever left in both artifacts, on either backend.

### Two backend bugs this export uncovered

Lowering cleanly was not enough: the first device run segfaulted, and two
distinct Vulkan backend bugs had to be fixed before LFM2.5 would run. Both are
generic — neither is specific to this model — and both are fixed on
`@ms/vulkan-mutable-buffers-and-index-gather` in software-mansion-labs/executorch.

**1. `aten.index.Tensor` was delegated past its own support check.**
`create_pattern_match_from_internal_match` put the target-graph nodes bound to a
PATTERN's placeholders into `PatternMatch.all_nodes`. Those nodes merely *feed*
the matched subgraph. That set becomes `VulkanPartitioner`'s `fusable_nodes`, and
`_is_node_supported` returns True for anything in it *before* consulting
`are_node_inputs_supported_fn` — so feeder nodes had their support checks
silently waived. RoPE's `freqs_cos[input_pos]` feeds the `hf_rope` pattern, so it
was delegated despite `check_index_tensor_node` rejecting its rank-2 `self`, and
the 1-D-only gather aborted in `virtual_resize`.

This is also why gemma4 works and this model did not: gemma4's pipeline never
emits `aten.index.Tensor` for RoPE, so it never touches the waiver.

Two obvious-looking escapes from this are both dead ends, so do not spend a day
rediscovering them. Letting the gather fall back to the host portable kernel
segfaults on a null buffer, because `freqs_cos` has already been prepacked into
the delegate. Splitting the delegate around it instead trips
`VulkanBackend.cpp:839 (args[o]->isTensor()) is false`, because the split pushes
a symint across the delegate boundary. The gather has to stay *inside* the
delegate, which is why the fix generalizes the op rather than declining it.

**2. Mutable buffers whose mutation is functionalized.** A mutated buffer reaches
`output` either as the buffer PLACEHOLDER (mutated in place, as
`sdpa_with_kv_cache` does to a KV cache) or as a COMPUTED node (functionalized
into new values). `is_mutable_buffer_node` only recognised the first. LFM2.5 has
both: 12 in-place KV caches and 10 short-conv states updated by `cat` +
`slice_copy`. The 10 computed ones were therefore

  * serialized into `output_ids`, which nothing passes to a delegate call, so
    `output_offset = args.size() - num_outputs` underflowed in `size_t` and every
    output access read a wild pointer — 11 declared outputs against a
    3-argument call, segfaulting in `MultimodalPrefiller::prefillChunk`; and
  * never copied back into the prepacked buffer tensor, so each short-conv
    recurrence restarted from zero on every call. This half is silent: the
    first call is correct either way.

The second bug is the one worth remembering when adding any recurrent
architecture here. It does not announce itself.

### Verified on device

Galaxy S26 Ultra (Adreno, driver 0842.19.8), `apps/llm` release build against a
locally built runtime with `EXECUTORCH_BUILD_VULKAN=ON`:

Both sizes generate correctly, text-only and with an image. Correct generation is
itself the load-bearing check: a broken short-conv copy-back degrades output
rather than erroring, so only a real multi-token run on real hardware can show it.

Baselines are the **published v0.9.0 XNNPACK artifacts**, i.e. what users get
today, not something re-exported for the comparison.

**Decode throughput** (image prompt, cool device, hand-read from the app):

| | xnnpack | vulkan | |
|---|---|---|---|
| 450M | 100 t/s | 150 t/s | **+50%** |
| 1.6B | 54 t/s | 72 t/s | **+33%** |

**TTFT, text-only** (no vision tower in the path, so this is pure decoder
prefill; two runs each, same prompt, driven over adb):

| | xnnpack | vulkan | |
|---|---|---|---|
| 450M | 326 / 323 ms | 210 / 160 ms | **~1.6x** |
| 1.6B | 559 / 572 ms | 301 / 307 ms | **~1.85x** |

TTFT is reproducible to within a few ms on the XNNPACK runs. Decode t/s is not:
the same runs measured 90-93 vs 108-139 (450M) and 48 vs 52-54 (1.6B) once
generations ran to 500-1900 tokens, because sustained load throttles the GPU
harder than the CPU and the KV cache grows. Short generations on a cool device
are the representative case; long ones compress Vulkan's advantage.

**Image TTFT is unchanged between backends, and that is expected**: with an image,
TTFT is vision tower + decoder prefill, and the tower is identical XNNPACK code
in both builds. At 1.6B the backend is worth ~260 ms of prefill, so a tower that
hides that difference is costing considerably more than 260 ms.

The delegate is version-locked to the shaders it was exported against: these were
built with `executorch 1.3.1+112df50` from software-mansion-labs/executorch, and
a PTE run against a different shader tree produces *gibberish rather than an
error*.

### Where the remaining TTFT is, if anyone wants it

The vision tower is the whole of image-prompt TTFT and is barely touched. What
the Vulkan partitioner would do with it (450M), versus XNNPACK today:

| | delegate calls | host ops |
|---|---|---|
| xnnpack (shipped) | 111 | 484 |
| vulkan | 39 | 168 |

Vulkan claims much more of it, but 39 delegate calls is still 39 CPU-GPU round
trips with staging copies at each boundary, so this is not obviously a win and
should be measured before it is believed.

The more promising target is *what* stays on the host, which is almost entirely
SigLIP2's attention-mask construction — 51 `expand_copy`, 24 `scalar_tensor`,
24 `mul.Scalar`, 24 `logical_not`, 12 `eq.Scalar`, 12 `full_like`, 12 `any.dim`.
That is padding-mask bookkeeping rather than real work, it is constant for a
fixed image size, and it is what shatters the tower on *both* backends. Folding
it away at export time would cut fragmentation far more than swapping backends.
