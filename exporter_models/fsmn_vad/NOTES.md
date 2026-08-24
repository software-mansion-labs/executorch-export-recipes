# fsmn_vad notes

Only XNNPACK is published. Both other backends have an exporter here and neither
ships; the HF repo carries `mlx/NOTES.md` and `coreml/NOTES.md` explaining why.

## MLX: exports well, fails at execute (measured 2026-08-21)

Unlike the convolutional vision models, fsmn_vad partitions cleanly into just
**3 subgraphs**. It still does not work: the `.pte` fails on device with
`Method 'forward' failed`.

The XNNPACK build is the control. Both configs declare an identical `forward`
contract, and the same harness fed both the same input on the same iPhone 16 in
the same run:

| backend | size | result |
| --- | --- | --- |
| **xnnpack fp32** | **1.83 MB** | **1.5 ms median, finite** |
| mlx fp32 | 2.87 MB | fails at execute |

MLX is 57% larger at the same precision, so it was never going to ship even if
it ran. The failure was not diagnosed further for that reason.

Two ops did not lower: `aten.as_strided_copy` (no handler), and `aten.clamp_`,
which hits a bug in the MLX clamp handler: it calls `float()` on a bound that
arrives as a `Slot` rather than a scalar, raising `TypeError`. The partitioner
catches it and downgrades the op to unsupported, so the export still succeeds.
Present on upstream `main` as well as the pinned 1.4.1 build.

## Core ML: not supported

`coreml/export.py` is entirely commented out and its body holds the writeup.
Core ML fails at execute with error `0x1` under `CPU_AND_NE` in both fp16 and
fp32, re-tested 2026-08-20 on a cleared `~/Library/Caches/executorchcoreml`
(a stale cache silently serves an older compilation and fakes a pass).
`CPU_ONLY` works but has no accelerator advantage over XNNPACK, so `coreml`
stays out of `model.json` backends. Related: apple/coremltools#2825.
