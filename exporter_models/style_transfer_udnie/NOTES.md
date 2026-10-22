# style_transfer_udnie notes

## MLX is NOT published (measured and rejected 2026-08-21)

`mlx/export.py` and `mlx/config.json` are kept so the decision can be re-tested,
and `mlx` stays in `model.json` backends so `--backend mlx` actually runs. No
`.pte` is published; the HF repo carries `mlx/NOTES.md` instead.

Measured on an iPhone 16 (ET 1.4.1, 3 warmup + 15 timed runs, median, inputs
allocated from the model's own schema and filled with 0.5). All four styles share
one architecture, so all four were run rather than generalising from one:

| style | coreml fp16 | mlx fp16 | penalty | mlx output |
| --- | --- | --- | --- | --- |
| candy | **193.4 ms** | 564.5 ms | 2.92x | non-finite |
| mosaic | **194.7 ms** | 546.4 ms | 2.81x | non-finite |
| rain princess | **193.3 ms** | 566.7 ms | 2.93x | non-finite |
| udnie | **196.9 ms** | 574.7 ms | 2.92x | non-finite |

**The output is the deciding fact, not the latency.** Every MLX build emits
NaN/Inf; Core ML fed the identical input in the same run returns finite output.
Size is no argument either: 3.48 MB (mlx fp16) vs 3.79 MB (coreml fp16).

**Root cause is op coverage.** The graph shatters into **50 subgraphs**:
`aten.flip` x64 (reflection pads), `aten._native_batch_norm_legit.no_stats` x15
(InstanceNorm), `aten.upsample_bilinear2d` x2. Note the overload: the MLX backend
registers `_native_batch_norm_legit_no_training` but not `no_stats`, and this
architecture is InstanceNorm in every block. Same class as the SDXS 28-subgraph
split, filed upstream as https://github.com/pytorch/executorch/issues/22017.
