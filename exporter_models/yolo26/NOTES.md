# yolo26 detection

## XNNPACK int8 (n, s at 640)

PT2E static, `get_symmetric_quantization_config(is_per_channel=True)`, 300 COCO
val2017 calibration images, with `model.0` (first conv) and `model.23` (Detect)
kept fp32. COCO mAP50-95 on the first 500 val2017 images, real `.pte`:

| size | fp32 | int8 | host ms fp32 -> int8 |
| --- | --- | --- | --- |
| n | 44.58 | 43.92 (-0.66) | 36.2 -> 28.1 |
| s | 50.67 | 50.06 (-0.61) | 89.5 -> 51.5 |

Quantizing Detect gives 0 mAP: its decode cats pixel boxes with sigmoid scores
under one int8 scale. ET 1.4.1 `_annotate_cat` ignores `filter_fn`, so
`set_module_type(Detect, None)` does not keep the cats fp32; the exporter strips
the annotations after the quantizer runs. 1000 calibration images did not help
(fake-quant n -0.85, s +0.21).

## Vulkan: shipped (fp16, Android)

`vulkan/export.py` drives `force_fp16` from the variant precision, matching
yolo26_seg. Only the 640 variant is declared, because 640 is the only resolution
measured; the earlier unmeasured 384 and 512 fp32 variants were dropped.

Vulkan fp16 against the published XNNPACK fp32, cold and thermally gated,
5 rounds with arm order reversed each round, per-exec slope from n=2 vs n=6,
every run validated:

| size | Adreno 840 (S26 Ultra) | Mali-G76 (S10+) | Adreno vk ms | Mali vk ms |
| --- | --- | --- | --- | --- |
| n | 1.16x (overlap) | 2.56x | 20.25 | 102.67 |
| s | **1.41x** | **3.51x** | 40.15 | 237.60 |
| m | **1.45x** | **3.66x** | 90.78 | 605.96 |
| l | **1.70x** | **3.85x** | 114.88 | 780.09 |
| x | **1.86x** | **1.98x sustained** | 238.37 | 3056 |

Vulkan wins at every size on both GPUs and the win grows with model size. Every
size partitions identically (**4 delegates, 5 CPU ops**), so nothing fragments as
the model grows and work per dispatch is the only variable. `n` is marginal on
Adreno (ranges overlap); ship `s` and larger.

Mali wins are ~2.5x larger than Adreno, and that is a CPU story: on yolo26n
Vulkan is 5.1x slower on Mali than Adreno (102.7 vs 20.3 ms) while XNNPACK is
11.2x slower (262.5 vs 23.4). The older device has a far weaker CPU baseline.

## The CPU tail is not worth implementing

Identical at every size:

    [DELEGATE] -> max.dim, topk -> [DELEGATE] -> topk, remainder -> [DELEGATE] -> index -> [DELEGATE]

`aten.max.dim` (1,8400,80), `aten.topk` x2, `aten.remainder.Scalar` and
`aten.index.Tensor` have no Vulkan implementation; `index.Tensor` additionally
carries **two** index tensors and the support check allows exactly one.

That whole tail was exported as a standalone portable-kernel `.pte` and timed on
device: **1.35 ms (1.19-1.46)**. That is 0.6% of yolo26x and 6.7% of yolo26n, so
removing it entirely could not take x past ~1.87x. `topk` is the hard op and it
gates 2 of the 3 breaks: the worst effort-to-payoff of the four.

## Measurement traps hit here

- **Gate on AP, not SKIN, on the S10+.** SKIN read 33.8 C / `mStatus=0` while AP
  sat at 47.8 C. The first Mali pass read l 2.21x, apparently breaking the size
  trend; re-gating on AP <= 42 C gave 3.85x and tightened the Vulkan arm from
  773-1297 ms to 732-789 ms.
- **yolo26x does not recover even when AP-gated.** It steps 1677 -> ~3050 ms
  after round 1 while its XNNPACK arm stays flat, so it is GPU-specific and not
  die temperature. Quote the sustained number: a 640px x-model is ~3 s/frame, so
  real use is sustained by definition.

## Export provenance

Exporting against stock `executorch==1.4.1` produces a **byte-identical** .pte to
the patched fork, so this model needs no fork pin. That is not true of every
Vulkan export in this repo: see the MR description for the models that do.
