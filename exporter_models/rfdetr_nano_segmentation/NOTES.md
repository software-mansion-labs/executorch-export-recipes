# rfdetr_nano_segmentation

## Vulkan: where the GPU time goes (Mali-G76, 2026-09-17)

Profiled with `ETVK_PROFILE=1` against a runner built from the same fork commit
the export pins. GPU total matched the RNE benchmark median (831 vs 831 ms), so
nothing is hiding on the CPU.

| kernel group | share |
|---|---|
| linear + matmul | 63% |
| view + permute (pure data movement) | 16% |
| masked-row guard | 4.5% |
| conv | 4% |

### Fixed: the masked-row guard (4.35%)

PyTorch's SDPA decomposition emits `aten._safe_softmax`, which appends a guard
that zeroes any row whose scores were all `-inf`. RF-DETR attends over a
fixed-size image with no padding and passes no mask, so it can never fire, but
the compare, two `logical_not`s, reduction, `zeros` and `where` all run over the
full attention matrix. `vulkan/export.py` overrides the decomposition.

The override must travel with the attention decomposition: SDPA is still a
single node after `run_decompositions`, so a table keyed on `_safe_softmax`
alone never fires.

Mali-G76, 6 interleaved rounds, median GPU: **811.1 -> 775.8 ms, 6/6 rounds won**.
Runner outputs bit-identical; detections on three COCO photos match the previous
build to 0.0000 px and 0.00000 confidence.

### Not fixed: no-op view copies (5.5% still on the table)

45 ms of `view_texture_half` is copies whose input and output map to the same
image extents, so every element keeps its texel: identical shapes (11 ms) and
rank-only changes such as `[677,384] -> [1,677,384]` (34 ms).

Only the identical-shape ones are safe to delete, and that is now done upstream
of here (`RemoveRedundantOpsTransform`). **Deleting the rank-changing ones is
not**: it was tried and the device aborted at `check_binary_op_args`, because
ops downstream read their argument's shape. Recovering those needs ET-VK to
adopt exir's `memory.view` / `_ViewSpec`, which keeps the view's shape while
sharing its base's storage through memory planning. Runtime aliasing via
`add_tensor_view` is NOT a shortcut: ET-VK tensors carry an AOT-assigned
`mem_obj_id`, so the planner may reuse the source while an alias still reads it.
