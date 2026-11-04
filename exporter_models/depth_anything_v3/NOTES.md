# Depth Anything 3 Small experiment

Source: `depth-anything/DA3-SMALL`, weights revision
`e08cab65ca0ec38e7826075418411ab90cab4da3`.
Upstream source: https://github.com/ByteDance-Seed/Depth-Anything-3
at `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
The minimal source subset is vendored in `_upstream/` with its Apache license.

Single-view depth-only wrapper: input `[1,3,518,518]`, output `[1,518,518]`,
float32. Camera pose, ray and confidence outputs are not part of this contract.
The upstream depth head is used unchanged; export dead-code elimination can
remove unused auxiliary computations. DA3 returns depth (larger means farther),
not V2's relative inverse depth. No reciprocal or display normalization is
applied by the wrapper.

For the controlled V2 comparison both use identical 518-square stretched RGB
inputs and ImageNet normalization. This differs from DA3's default
aspect-preserving input processor; comparisons are for this deployment policy.

XNNPACK FP32 first; an optional INT8 variant uses dynamic activation/per-channel
weight quantization of linear layers only, matching the V2 recipe. Export and
runtime results will be recorded below. No Hugging Face publication requested.

## Export adaptations

- The official checkpoint stores a shared auxiliary LayerNorm only once. The
  wrapper restores its aliases before strict loading; unrelated missing or
  unexpected weights still fail.
- RoPE derives its lookup extent from the fixed patch grid instead of
  `int(positions.max())`, which torch.export cannot capture. Index tensors stay
  int64 through PT2E preparation.
- Small image-independent tensors are folded before PT2E; large spatial
  embeddings remain broadcast expressions to avoid bloating the program.
  Folded tensors use contiguous strides. The generic Torch folding pass in
  this pinned stack references an unavailable `pt2e_quant.dequantize_affine`;
  the local bounded pass avoids it. INT8 conversion uses `fold_quantize=False`
  followed by the local pass, including quantized weights.

## Reproduce

```sh
uv run --project exporter_models/depth_anything_v3 exporter export depth_anything_v3 \
  --size small --backend xnnpack --variant fp32 --output-dir /private/tmp/depth_anything_v3/xnnpack
uv run --project exporter_models/depth_anything_v3 exporter export depth_anything_v3 \
  --size small --backend xnnpack --variant int8 --output-dir /private/tmp/depth_anything_v3/xnnpack
uv run --project exporter_models/depth_anything_v3 python exporter_models/depth_anything_v3/compare.py \
  --v2-dir /private/tmp/depth_anything_v2 --v3-dir /private/tmp/depth_anything_v3 \
  --image /path/to/photo.jpg --output-dir /private/tmp/depth_anything_v3/comparison --runs 15
```

Repeat `--image` to compare more images. `--v2-dir` expects `xnnpack/` FP32 and
`optimized/` INT8 V2 files; `--v3-dir` expects `xnnpack/` V3 files. The comparison
checks the serialized float32 IO contract and XNNPACK delegates, executes the
programs, checks V3 against its unmodified PyTorch depth path, writes a PNG
contact sheet and raw `.pt` results, then benchmarks all available variants in
rotating order. It fails on FP32 range-normalized RMSE above 0.01%.

## Local results (2026-10-19)

Apple M1 Pro, four threads, fixed 518-square inputs. Median of 15 native runs
per variant after two warmups. Exports were finished before timing. These
measurements exclude image preprocessing and rendering. The V2 artifacts are
the published V2 FP32 and attention-optimized INT8 variants.

| Variant | File size | Median | XNNPACK partitions | Portable calls |
| --- | --- | --- | --- | --- |
| v2_fp32 | 99.0 MB | 501.9 ms | 50 | 79 |
| v2_int8 | 35.5 MB | 308.8 ms | 38 | 30 |
| v3_fp32 | 106.5 MB | 739.1 ms | 65 | 282 |
| v3_int8 | 40.7 MB | 585.7 ms | 65 | 282 |

V3 FP32 matched eager on all seven inputs; largest absolute error was
1.55e-05. V3 INT8's worst range-normalized RMSE was 1.44% and
its lowest correlation with eager was 0.99803. Quantization thresholds
are provisional numerical checks, not measured depth accuracy.

Images: the existing cats photo, the two official DA3 SOH photos, and V2's
held-out `demo17`–`demo20` illustrations. The contact sheet displays reciprocal
V3 depth to give both versions a common near-bright polarity. Each map uses
its own 2nd–98th percentile display limits. Raw values are saved unchanged.

On this small sample, V3 produces smoother maps and often softer object
boundaries than V2. There is no ground-truth depth, so this is a visual
comparison, not evidence that either model is more accurate. V3's native
latency and portable-call count are higher in this first export; V2 remains
the better measured latency choice for the current single-image task.

Artifacts and detailed results:
- `/private/tmp/depth_anything_v3/xnnpack/`
- `/private/tmp/depth_anything_v3/comparison/comparison.json`
- `/private/tmp/depth_anything_v3/comparison/v2-v3-depth-comparison.png`

## CoreML comparison (2026-10-19)

CoreML FP16, minimum deployment target iOS 18, `ComputeUnit.ALL` for both
models. Same Apple M1 Pro and seven saved 518-square inputs as above. The
comparison uses the native ExecuTorch CoreML delegate. First-load compilation
is excluded; timings are medians of 15 alternating calls after two warmups.

| Model | File size | Median | CoreML partitions | Portable calls |
| --- | --- | --- | --- | --- |
| v2_coreml_fp16 | 50.2 MB | 57.9 ms | 1 | 0 |
| v3_coreml_fp16 | 70.9 MB | 65.4 ms | 1 | 0 |

v2_coreml_fp16: worst range-normalized RMSE 0.806%, minimum correlation 0.999906.
v3_coreml_fp16: worst range-normalized RMSE 0.408%, minimum correlation 0.999933.

All numerical checks passed: True. V2 is compared against its validated
XNNPACK FP32 output, V3 against unmodified eager PyTorch. This checks export
parity, not depth-estimation accuracy. These are macOS measurements.

Reproduce after generating the XNNPACK comparison references:

```sh
uv run --project exporter_models/depth_anything_v3 exporter export depth_anything_v3 \
  --size small --backend coreml --variant fp16 --output-dir /private/tmp/depth_anything_v3/coreml
uv run --project exporter_models/depth_anything_v3 python exporter_models/depth_anything_v3/compare_coreml.py \
  --v2-model /private/tmp/depth_anything_v2/coreml/depth_anything_v2_small_coreml_fp16.pte \
  --v3-model /private/tmp/depth_anything_v3/coreml/depth_anything_v3_small_coreml_fp16.pte \
  --references /private/tmp/depth_anything_v3/comparison \
  --output-dir /private/tmp/depth_anything_v3/coreml_comparison --runs 15
```

Reports and the visual comparison are saved under
`/private/tmp/depth_anything_v3/coreml_comparison/`.

## Vulkan and publication

Vulkan FP16 export: 56,942,082 bytes, one delegate, no portable operations.
Static RoPE constants are folded and attention uses explicit matmul/softmax,
avoiding the fused SDPA serializer's non-finite scalar. Seven adapted PyTorch
inputs match the original float model (atol/rtol 5e-5). Maximum 3D texture extents
[1369,1370,1370] fit the 2048 guard. Android runtime execution remains pending.
Reports: `/private/tmp/depth_anything_v3/vulkan_verification.json` and
`vulkan_pytorch_parity.json`.

Published all four variants using Git LFS to
https://huggingface.co/software-mansion/react-native-executorch-depth-anything-v3
at commit `3b2368a`, with release tag `v0.11.0`.

### Corrected Vulkan output arity

The initial artifact serialized twelve positional constant buffer mutations as
Vulkan outputs, while the outer delegate call passed only the image and depth
map. Android correctly rejected it with `args.size() == num_inputs + num_outputs`.
The export now functionalizes the graph before promoting constant computations
to buffers. `verify_delegate_arity` checks the serialized runtime boundary and
rejects the original artifact (2 arguments versus 1 input + 13 outputs). The
replacement has 2 arguments, 1 input and 1 output; no buffer mutations remain.

Corrected artifact: 56,407,810 bytes, SHA256
`b3b22b7ca70539010d4213ec9f1b03e4de492793f4ec36fd846a90266e664203`.
Published via Git LFS in commit `dfe044b`; `v0.11.0` updated with an explicit
lease against the old tag. Re-download cached copies of the old artifact.
`verify_vulkan.py` checks seven saved float references and repeat execution;
`vulkan/verify_android.cpp` is the small Android runner used for device checks.

### Android execution

Corrected Vulkan artifact executed on Samsung SM-G986B (Galaxy S20+), Android 13,
using the current React Native ExecuTorch shared libraries. Three inputs
(cats, SOH/000, demo20), three executions each: all finite, correct float32 shape.
Worst range-normalized RMSE 0.2017%, minimum correlation 0.9999728 against original
float references. Warm runs approximately 1.10–1.11 seconds.
Report: `/private/tmp/depth_anything_v3/vulkan_android_verification.json`.
This supersedes the earlier note that Android execution was pending.

## XNNPACK attention cleanup (2026-10-20)

For the static singleton-view contract, replace SDPA with explicit three-dimensional
batched matmul/softmax attention and slice Q/K/V along channels. This removes
SDPA's generic masking checks and broadcast copies. Functionalize positional
arithmetic before bounded constant folding, eliminating constant-buffer mutations.
No upstream vendored source, checkpoint, output polarity, or public IO changed.

Both FP32 and INT8 now have 37 XNNPACK delegate calls and 106 portable calls,
down from 65/282. Remaining portable calls include 48 native layer norms, 22
unsqueezes, 16 splits, and smaller shape/selection operations. Counts are runtime
call boundaries, not a percentage of original model operators.

Four-thread M1 Pro paired benchmark, three images, two warmups per image followed
by five alternating old/new runs (15 timed samples per variant):

| Precision | Baseline median | Candidate median | Speedup |
|---|---:|---:|---:|
| FP32 | 753.08 ms | 532.15 ms | 1.42x |
| INT8 | 596.94 ms | 374.51 ms | 1.59x |

Seven saved original references passed. FP32 maximum range-normalized RMSE
0.000116%; zero/random inputs also pass atol/rtol 5e-5. Repeated execution is
identical. INT8 worst normalized RMSE is 1.50% (previous 1.44%), minimum
correlation 0.998355; still within the provisional 2% / 0.99 check. These are
numerical comparisons, not task accuracy measurements.

Reproduce baseline/candidate checks with `tools/compare_depth_xnnpack.py --help`.
Artifacts and full results: `/private/tmp/depth-optimized/v3/xnnpack`,
`/private/tmp/depth-optimized/comparison.json`, and `stress.json`.
No ETDump per-op timing is claimed: the installed wheel lacks event tracing.

Published the verified XNNPACK update via Git LFS at `f9295ac`, main and
`v0.11.0`. Delete cached prior XNNPACK files before re-downloading.
