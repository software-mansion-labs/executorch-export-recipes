# qwen3

## Config is vendored, not fetched

The published `{size}/xnnpack/config.json` got two things wrong for every size,
so a corrected copy lives here:

- `get_n_layers: null` where the `.pte` carries the real count (28 / 28 / 36).
- the `forward` sequence dim declared `max: 2048` where the `.pte` declares
  2047 (`max_seq_len - 1`).

Both are repo-wide across the LLM families, not specific to Qwen3. Everything
else matched.

## Published artifacts

All six `.pte` match this config. Verified with:

```bash
uv run --project exporter_models/qwen3 python tools/probe_published_pte.py qwen3 --size 0_6b
```

Nothing to re-export.

## Vulkan: exporter kept, nothing published

`vulkan/export.py` and `0_6b/vulkan/config.json` produce a working, verified
0.6B 8da4w artifact, but no `.pte` is published. The measurements below are the
reason: the backend wins decisively on one axis and one GPU, and loses on the
others, so there is no single artifact worth shipping yet. `vulkan` stays in
`model.json` backends so `--backend vulkan` runs and the finding can be
re-tested.

### The recipe is three settings, and each one matters

1. **A plain `VulkanPartitioner`, attention included.** Keeping
   `llama::custom_sdpa` and `llama::update_cache` on the host splits the decoder
   into 57 subgraphs, two delegate calls plus three portable kernels per layer,
   and every boundary costs a staging copy and a queue submission. Letting
   Vulkan's `causal_sdpa` pattern fuse them back collapses the model to a
   **single delegate**: decode 41.5 to 86.2 tok/s on an Adreno 840, and RSS from
   768 to 300 MiB because the KV cache moves into GPU memory. There is no
   Adreno SIGSEGV, contrary to what the LFM2.5-VL recipe reports for its sizes.
2. **`enable_dynamic_shape = True`.** ExecuTorch's `TextPrefiller` prefills one
   token at a time whenever this is false, so a static-shape export measures the
   same prefill rate as its decode rate. This is the whole prefill story.
3. **`force_fp16`.** Same generated text, half the KV traffic. Worth nothing at
   short context and ~15% of decode at a couple of hundred tokens, so it has to
   be measured deep.

### Results

Galaxy S26 Ultra (Adreno 840), 206-token prompt, 3 runs plus an order-reversed
pass, against the published XNNPACK 8da4w:

| backend | prefill tok/s | decode tok/s | RSS | file |
| --- | --- | --- | --- | --- |
| Vulkan | **1685** | 71 | **303 MiB** | 582 MB |
| XNNPACK | 627 | **82** | 1088 MiB | 506 MB |

2.7x prefill, 0.86x decode. Time to first token is 93 ms against 344 ms. Still
holds at 773 prompt tokens (1484 against 598 prefill).

Galaxy S10+ (Mali-G76) inverts it, same artifact:

| backend | prefill tok/s | decode tok/s | RSS |
| --- | --- | --- | --- |
| Vulkan | 11.6 | **12.0** | 1038 MiB |
| XNNPACK | **52.6** | 7.2 | 1028 MiB |

The decode win grows with context, because Vulkan decode is nearly flat while
XNNPACK collapses: 83.8 to 97.1 ms/token from a 19- to a 773-token prompt,
against 59.4 to 336 ms/token. At 773 tokens Vulkan decodes 3.4x faster. The
crossover is somewhere around 100 to 200 tokens of context.

Prefill is the problem on Mali, and it has a clear signature: prefill costs
about the same per token as decode there, so batching the prompt buys almost
nothing, against a 24x batching benefit on Adreno. That points at the int4
linear kernel not tiling over the sequence dimension on Mali. The memory win
does not transfer either, since Mali's allocations are host-visible and show up
in RSS.

### `force_fp16` is Adreno-only

Measured forward and order-reversed on the Mali device at 206 tokens: fp32
prefills at 16.5 tok/s against fp16's 11.6, and decodes at 13.0 against 12.0. On
Adreno fp16 is worth 14% of decode and nothing on prefill. So a single fp32
artifact would cost 14% of Adreno decode and buy 42% of Mali prefill; fp16 is
44 MB smaller. The exporter keeps `force_fp16` because that is the
configuration the Adreno numbers above were taken on, but fp32 is the better
choice if this is ever published as one file.

### Two ExecuTorch fixes this needed

- `graph_output_allocated not set` from the memory planner, which is what blocks
  dynamic shapes: pytorch/executorch#22477, PR #22479. **Required** to reproduce
  the export. Until it lands, `uv run --project exporter_models/qwen3 exporter
  export qwen3 --backend vulkan` fails on that assert, because this dir pins
  executorch to `software-mansion-labs/executorch @ms/separate-backends-1.4.1`
  and the fix is not on that branch yet. The numbers above were taken with the
  fix applied to a local checkout of that same branch.
- A Vulkan delegate that declares fewer inputs than the call passes, which is
  what blocked the host-attention variant: pytorch/executorch#22478, PR #22480.
  Not needed for the recipe above.
