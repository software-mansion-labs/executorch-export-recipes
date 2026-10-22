# fast_sam notes

## Core ML: pinned to CPU_AND_NE (measured 2026-08-24)

`compute_unit=ct.ComputeUnit.CPU_AND_NE`. ExecuTorch defaults to `ComputeUnit.ALL`,
which here splits the graph across GPU and ANE and pays the transitions. iPhone 16,
ET 1.4.1, 10 warmup + 100 timed, median of two interleaved reps:

| variant | published (ALL) | pinned CPU_AND_NE | |
| --- | --- | --- | --- |
| fast_sam s | 52.45 ms | **8.70 ms** | 6.0x |
| fast_sam x | 108.85 ms | **38.90 ms** | 2.8x |

**This is not a general rule.** A full four-way panel (ALL / CPU_ONLY /
CPU_AND_GPU / CPU_AND_NE) over eleven Core ML models found `ALL` already optimal
for efficientnet, ssdlite, fcn, selfie and yolo26, and `CPU_AND_GPU` worse than
`ALL` almost everywhere (fcn r50 47.7 -> 210.8 ms). Only fast_sam and style
transfer justified a pin, and they wanted *different* units: style transfer is
pathological on the ANE and ships CPU_AND_GPU, fast_sam is the opposite.

Two traps this panel produced, both worth repeating:

- **15 timed runs is not enough.** lraspp looked like a 1.2x win for CPU_AND_NE
  at 15 runs (8.3 vs 7.2 ms) and was 5.90 vs 5.90 ms at 200 runs with 10 warmup.
  The apparent win was warmup, not placement.
- **Compare the same variant.** deeplab appeared to gain 4.0x, but the sweep
  exported the model's *default* variant (mobilenet_v3_large) while the "ALL"
  baseline downloaded was resnet50. That measured two backbones, not two compute
  units. Like-for-like, deeplab gains nothing (0.98-1.06x) and is not pinned.

Verify a Core ML rebuild by measuring the exact artifact you intend to publish
against the exact artifact it replaces; Core ML export is not byte-reproducible,
so hashing cannot confirm a rebuild.

