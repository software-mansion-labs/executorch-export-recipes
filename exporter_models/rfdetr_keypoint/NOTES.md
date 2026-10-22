# rfdetr_keypoint

## Vulkan: where the GPU time goes (Mali-G76, 2026-09-17)

Profiled with `ETVK_PROFILE=1` against a runner built from the same fork commit
the export pins. GPU total matched the RNE benchmark median (3177 vs 3244 ms),
so nothing is hiding on the CPU.

| kernel group | share |
|---|---|
| linear + matmul | 59% |
| view + permute (pure data movement) | 19% |
| masked-row guard | 6.3% |
| `sum_per_row` | 4.6% |
| conv | 3% |

The graph is well partitioned: 2 delegate calls and 3 CPU kernels (`max`,
`topk`, `scalar_tensor`). The old 34-to-39 delegate fragmentation is gone.

### Fixed: the masked-row guard (5.96%)

PyTorch's SDPA decomposition emits `aten._safe_softmax`, which appends a guard
that zeroes any row whose scores were all `-inf`. RF-DETR attends over a
fixed-size image with no padding and passes no mask, so it can never fire, but
the compare, two `logical_not`s, reduction, `zeros` and `where` all run over the
full attention matrix. Per attention block that measured 9.8 ms against 28.5 ms
of actual matmul. `vulkan/export.py` overrides the decomposition.

The override must travel with the attention decomposition: SDPA is still a
single node after `run_decompositions`, so a table keyed on `_safe_softmax`
alone never fires.

Mali-G76, 5 interleaved rounds, median GPU: **3162.7 -> 2974.3 ms, 5/5 rounds
won**, 174 dispatches removed. Runner outputs bit-identical; detections and
landmarks on three COCO photos match the previous build to 0.0000 px.

### Not fixed: no-op view copies (7.8% still on the table)

247 ms of `view_texture_half` is copies whose input and output map to the same
image extents, so every element keeps its texel: identical shapes (64 ms),
rank-only changes like `[677,384] -> [1,677,384]` (72 ms), and leading-dim
merges like `[4,6,577,64] -> [24,577,64]` (112 ms).

Only the identical-shape ones are safe to delete, and that is now done upstream
of here (`RemoveRedundantOpsTransform`). **Deleting the others is not**: it was
tried and the device aborted at `check_binary_op_args`, because ops downstream
read their argument's shape. Recovering them needs ET-VK to adopt exir's
`memory.view` / `_ViewSpec`, which keeps the view's shape while sharing its
base's storage through memory planning. Runtime aliasing via `add_tensor_view`
is NOT a shortcut: ET-VK tensors carry an AOT-assigned `mem_obj_id`, so the
planner may reuse the source while an alias still reads it.

### Unexamined

`sum_per_row_buffer_half` at `{1, 435200, 1}` costs 34 ms per call, 4 calls,
4.6% of total. 64 threads per row over 435200 rows. Not investigated.
