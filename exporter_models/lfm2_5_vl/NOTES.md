# lfm2_5_vl notes

LFM2.5-VL vision-language model (vision_encoder + token_embedding + text_decoder),
sized `[450m, 1_6b]`, exported via the vendored `_lfm2_5_vl/` package (model code
from the executorch `examples/models/lfm2_5_vl` recipe; shared llama
transformer/kv-cache/sdpa infra kept as an import). One universal `export.py` per
backend reads `config["size"]` and maps it to `LiquidAI/LFM2.5-VL-{450M,1.6B}`.

## Why local config.json (not an HF pointer)

The published `react-native-executorch-lfm-2.5` repo does not ship a
`{backend}/{size}/config.json` for the VL sizes, so the multi-method VL contract
is hand-authored locally per `{backend}/{size}/`. 1.6B shapes scale with the text
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

## Core ML: not supported, and not attempted

This model has **no Core ML export and is not expected to get one.** Apple-silicon
acceleration here is MLX only (see `mlx/export.py`), which additionally requires the
software-mansion-labs ExecuTorch fork -- the PyPI wheel produces an MLX payload the
RNE runtime cannot load.

**Why Core ML is not on the table.** This is an autoregressive decoder driven by a
mutable KV cache. ExecuTorch's Core ML backend has no practical path for that: the
cache is a mutable buffer carried across calls, and the decode step needs a growing
sequence dim. Core ML's dynamic-shape support is the exact area where we already
have a confirmed, filed defect -- non-CPU compute paths reject shapes Core ML infers
as data-dependent (apple/coremltools#2825, see `docs/coremltools-2825-repro.md`).

**Status: a judgment call, not a measured result.** Nobody has run this model through
the Core ML backend and collected a failure. It is not attempted because the cost is
high and the prior is bad, not because it was tried and rejected. If Core ML gains
usable stateful-KV-cache support, this is worth revisiting.

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

The delegate is version-locked to the shaders it was exported against, and a PTE
run against a different shader tree produces *gibberish rather than an error*.

The runtime to build against is **`@ms/separate-backends` (05afc5d) on
software-mansion-labs/executorch**, which carries the two Vulkan fixes this
export cannot run without:

    abcfb9f  Fix aten.index.Tensor being delegated past its own support check
    05afc5d  Fix mutable buffers whose mutation is functionalized

The published PTEs were exported from a `112df50`-based checkout rather than
this one, so for the avoidance of doubt: `backends/vulkan/` is byte-identical
between the two (`git diff 085b767 05afc5d -- backends/vulkan/` is empty), and
the artifacts on the Hub are valid against the split branch.

Without those fixes the failure is not graceful. The mutable-buffer bug is a
SIGSEGV on a stock runtime -- the arg-count guard that turns it into a named
error is itself part of the first fix.

### Published artifacts

On the Hub under `main` (the v0.10.0 tag is cut at release):

| | path |
|---|---|
| 450M | `vl_450m/vulkan/lfm_2_5_vl_450m_vulkan_8da4w.pte` |
| 1.6B | `vl_1_6b/vulkan/lfm_2_5_vl_1_6b_vulkan_8da4w.pte` |

Both are the bmm builds. Each ships a `config.json` whose method shapes were
read out of the PTE rather than inferred -- note `get_max_seq_len` is 256
(`PREFILL_SEQ_CAP`) against a 2048 context, and that `token_embedding` is *not*
capped, it stays at 2048 while only `text_decoder` takes the cap.

### The vision tower's attention mask, and why image TTFT was flat

`encode_images` passes an all-ones `pixel_attention_mask` — every patch of a
512x512 image is valid — so the tower's attention mask is uniformly permissive
and computing it is pure overhead. Removing it is **bit-identical**.

Removing it alone is nearly worthless, though: it deletes 5 host ops. The
expensive half is that torch's SDPA decomposition emits a fully-masked-row NaN
guard (`eq`/`any`/`logical_not`/`full_like`/`mul`, per layer) *whether or not a
mask is present*. Both have to go: call the encoder with `attention_mask=None`
**and** use `_attn_implementation="eager"`, which for an unmasked encoder is the
same matmul/softmax/matmul but emits no guard and partitions whole.

That still leaves 48 host ops, and they are all `expand_copy`. Edge dialect
lowers a 4-D `matmul` to `expand_copy` + `bmm`, and XNNPACK does not claim
`expand_copy`; both operands already have the output shape here, so each expand
is a pure ~3 MB copy that computes nothing (2 matmuls x 2 operands x 12 layers).
Folding the head dim into the batch dim and calling `bmm` directly leaves
nothing to broadcast.

Counted in the shipped PTEs, not in the exporter, so these are what actually
ran (`vision_encoder` method, delegate calls / host kernel calls):

| vision tower | 450M | 1.6B |
|---|---|---|
| xnnpack, as shipped | 51 / 229 | — |
| xnnpack, unmasked eager | 51 / 75 | 111 / 165 |
| xnnpack, + bmm attention | **27 / 27** | **57 / 57** |
| vulkan, unmasked eager | 27 / 52 | — |

The host remainder after bmm is 25 (450M) / 55 (1.6B) `native_layer_norm` plus
2 `unfold_copy`; no `expand_copy` survives.

On device (Galaxy S26 Ultra, same fixed image and prompt). **Only the first
prompt after a load is meaningful**: later prompts in the same conversation all
land near 300 ms whatever the build, because the tower cost is already paid, so
a same-conversation series measures nothing about the tower.

| image TTFT, first prompt after load | 450M | 1.6B |
|---|---|---|
| as shipped | ~1006 ms | — |
| unmasked eager | ~800 ms | 2133 ms |
| + bmm attention | **~661 ms** | **1872 ms** |
| unmasked eager, tower on vulkan | ~1800 ms | — |

Both steps are real and both scale with what they delete. Dropping the mask and
the guard is worth ~200 ms at 450M; the bmm rewrite is worth ~140 ms at 450M
(48 `expand_copy`) and ~260 ms at 1.6B (108 of them). Together, -34% at 450M.
Decode is unchanged (177 -> 180 tok/s), as expected — only the tower moved.

Measurement notes, both of which produced wrong numbers here first:

- Fresh-load each build. An earlier pass reported 1001/939 -> 711/709 and read
  as a flat ~27% win; those were four cold runs and say nothing about the warm
  path, which never moves.
- Do not measure a PTE you just pushed. The first load after a 2.5 GB push cost
  +300 ms of page-in and made the 1.6B bmm build look *slower* than the build it
  beats by 260 ms.

Numerics: dropping the mask is bit-identical, and so is the bmm rewrite. Eager
attention reassociates the reduction and moves the tower output by 1.5e-6 mean
absolute against a signal of magnitude 0.19 (7.6e-6 relative), orders of
magnitude below the 8da4w quantization error this export already accepts.

**The XNNPACK export gets the same 229 -> 27 reduction** and is not currently
taking it. Worth applying there too, which would speed up the published model.

### Where the remaining TTFT is, if anyone wants it

The vision tower is the whole of image-prompt TTFT and is barely touched. What
the Vulkan partitioner would do with it (450M), versus XNNPACK today:

| | delegate calls | host ops |
|---|---|---|
| xnnpack (shipped) | 111 | 484 |
| vulkan | 39 | 168 |

Vulkan claims much more of it, but 27 delegate calls is still 27 CPU-GPU round
trips with staging copies at each boundary. Measured, it is 2.25x slower
(~1800 ms vs ~800 ms for the same unmasked-eager tower on XNNPACK), so the
tower stays on XNNPACK. The artifact is kept as `vk450m_tower` so that stays
falsifiable.

What remains on host after the changes above is 25 (450M) / 55 (1.6B)
`native_layer_norm` plus 2 `unfold_copy`.

### Four things that look like wins and are not

All four were built and measured. None shipped. Recorded so nobody spends the
day on them twice.

**Give XNNPACK the layer norms.** XNNPACK has no layer-norm operator at all --
not merely no partitioner config, the library has none -- so all 25/55 of them
split the tower, and it looked like the obvious next target. They can be
eliminated: write the norm as mean/sub/mul/rsqrt, and substitute a matmul
against a constant 1/D vector for the mean, because `mean.dim` looks claimable
but only covers spatial reduction of a 4-D tensor (decomposing naively is
*worse*, 52 delegate calls instead of 27). The tower then collapses to **2
delegate calls / 2 host ops** at both sizes -- 25 and 55 boundaries gone.

It changes nothing measurable. Not at 450M, and not at 1.6B either, where there
are more than twice as many. Partition count is not what costs time here; the
rewrite also trades those boundaries for extra GEMV arithmetic inside the
delegate, and the two evidently cancel. This is the useful negative result of
the batch: **the tower is arithmetic-bound, not overhead-bound**, so the only
remaining levers are ones that do less arithmetic.

**Quantize the tower.** It is the last unquantized part of the model (decoder
8da4w, embeddings 4-bit, tower fp32), so it looks like the biggest lever left.
It is not reachable:

| | tower partitioning | relative error |
|---|---|---|
| fp16 | 102 / 176 (worse) | — |
| int8 static, whole tower | 78 / 92 (worse) | 184% |
| int8 static, linears only | 2 / 2 | 84% |
| int8 dynamic, linears only | 14 / 2 | 89% |

XNNPACK declines fp16 GEMMs outright and drops all 122 to host. int8 partitions
fine when confined to the linears, but the numerics do not survive, calibrated
on six real photographs with per-channel weights -- the SigLIP-family activation
outlier problem. It would need outlier-aware quantization or a hand-picked fp32
keep-list, which is a project, not a recipe change.

**Fewer visual tokens.** 384x384 (576 patches -> 144 tokens instead of 256)
would cut tower work and decoder prefill together. It does not export: SigLIP2's
position table is a fixed 1024 rows and this inlined encode path does not
interpolate it (`broadcast [1024, 768] vs [1, 576, 768]`). Fixable, but it also
needs react-native-executorch's `encoderTokenCount` to stop being a constant, so
it is an export *and* runtime change.

**Alias the mutable-buffer mutations.** Upstream's `alias_buffer_mutations`
would drop the copy-back this export's Vulkan fix relies on, saving 10 shader
dispatches per token. It does not exist in the ET tree this is built against and
would need runtime support as well; given the layer-norm result, dispatch count
is unlikely to be worth the port.
