# style_transfer_candy notes

## Core ML: pin the compute unit, the ANE is a trap here (measured 2026-08-24)

`compute_unit=CPU_AND_GPU` in every style's `coreml/export.py`. Measured on an
iPhone 16 (ET 1.4.1, 3 warmup + 15 timed, median, mean over the four styles):

| config | mean | note |
| --- | --- | --- |
| fp16 `ALL` | 209.4 ms | ANE-eligible, and the slowest |
| fp16 `CPU_ONLY` | 109.7 ms | fastest mean, but 99.6-119.9 ms spread |
| **fp16 `CPU_AND_GPU`** | **114.2 ms** | shipped; ~114 ms on all four styles |
| fp32 `ALL` | 152.8 ms | dropped at v0.10.0 |

ExecuTorch defaults to `ComputeUnit.ALL`, under which fp16 is ANE-eligible and
fp32 is not (the ANE is fp16-only). fp32 therefore *looked* faster than fp16 and
its removal was contested on those grounds
([#1375](https://github.com/software-mansion/react-native-executorch/pull/1375)).
The comparison was measuring placement, not precision: exclude the ANE and fp16
is the fastest build at half of fp32's size. Same op mix that shatters the MLX
partition into 50 subgraphs (InstanceNorm every block, reflection pads built from
`flip`/`cat`) appears to fragment the ANE path too.

`CPU_AND_GPU` over `CPU_ONLY` despite the slightly worse mean: `CPU_ONLY` ranges
99.6-119.9 ms across the four styles while `CPU_AND_GPU` holds ~114 ms on all
four, so it should degrade less under CPU contention and thermal load.

**Core ML export is NOT byte-reproducible.** Two exports from identical code
differ in sha256 at identical size (UUIDs embedded in the mlmodelc blob), so the
byte-identity check used for XNNPACK does not transfer. Verify a Core ML rebuild
by measuring it on device, not by hashing it.

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
