# ZipDepth base experiment

Upstream: https://github.com/fabiotosi92/ZipDepth
Source and NPU checkpoint revision: `91f3fd21e131641f51e8d35736d1958350180e3a`.
Checkpoint SHA256: `627c04fda584133ead4310074884a4a037061b4c01ba86e73e492ea30fab570d`.
Minimal upstream source and its MIT license are vendored in `_upstream/`.

Uses `base`, `global_mode=balanced`, `upsample_unfold=False`, strict weight loading,
and the upstream RepVGG / Conv-BN inference fusion. Float32 RGB input
`[1,3,384,384]` in `[0,1]`; ImageNet normalization is INSIDE the model.
Do not apply IMAGENET_NORM outside it. The output is a raw float32 map
`[1,384,384]`. No display normalization is embedded.

384 is upstream's default input size and differs from our V2 exports' 518 input.
The comparison will state both resolutions explicitly and resize outputs for
visual inspection. This is a deployment comparison, not equal-resolution accuracy.

## Export and local validation

ExecuTorch 1.4.1. Run from the exporter root using this model's pinned environment:

```sh
uv sync --project exporter_models/zipdepth
PYTHONPATH=src exporter_models/zipdepth/.venv/bin/python -m exporter.cli export zipdepth --size base --backend xnnpack --output-dir /private/tmp/zipdepth/xnnpack
PYTHONPATH=src exporter_models/zipdepth/.venv/bin/python -m exporter.cli export zipdepth --size base --backend coreml --output-dir /private/tmp/zipdepth/coreml
PYTHONPATH=src exporter_models/zipdepth/.venv/bin/python -m exporter.cli export zipdepth --size base --backend vulkan --output-dir /private/tmp/zipdepth/vulkan
```

- XNNPACK FP32: 24.6 MB; eager and fused references matched on seven photos/illustrations.
- CoreML FP32: 24,858,521 bytes, one delegate and no portable operations. Seven inputs passed; worst range-normalized RMSE 0.000116%, minimum correlation 0.99999976. Median 5.71 ms (15 warm runs, M1 Pro).
- Vulkan FP16: 12,341,250 bytes, two delegates and one portable `aten::_adaptive_avg_pool2d.out` operation. Float32 public IO verified. Maximum texture extents [576,576,96], below 2048. Android execution pending.
- CoreML FP16 was rejected: up to 6.31% range-normalized RMSE and correlation 0.970. It is excluded from published configuration. The export function still supports the precision for experiments.

Local reports: `/private/tmp/zipdepth/{backend_verification,vulkan_verification}.json`.
The saved-photo check can be repeated with `verify_backends.py` after the V3 comparison inputs exist under `/private/tmp/depth_anything_v3/comparison`.

## Deployment comparison

M1 Pro, four CPU threads, 15 interleaved warm runs: V2 INT8 (518) 330.25 ms;
ZipDepth FP32 (384) 27.79 ms; experimental ZipDepth FP32 (512) 48.35 ms.

Small NYUv2 probe: first 12 validation rows from `tanganke/nyuv2`, inverse-depth
least-squares scale/shift alignment per image, valid GT 0.1–10 m, 5% border
excluded. This is a deployment comparison, not an official benchmark.

| Model | AbsRel | RMSE (m) | delta1 |
|---|---:|---:|---:|
| V2 FP32 518 | 0.04496 | 0.28785 | 0.98423 |
| ZipDepth FP32 384 | 0.06622 | 0.37939 | 0.97114 |
| ZipDepth FP32 512 (experiment) | 0.07149 | 0.38034 | 0.95249 |

The 384 export is the selected fast preset. Increasing resolution did not improve
this probe. ZipDepth preserves broad geometry but has softer boundaries and loses
some illustrated-object detail. Inputs have different deployment resolutions and
interpolation policies (V2 bicubic 518, ZipDepth bilinear 384).

Reproduce the seven-image and labeled probe using `compare.py --artifacts
/private/tmp/zipdepth --references /private/tmp/depth_anything_v3/comparison
--nyu-dir /private/tmp/zipdepth/nyu_probe --output-dir /private/tmp/zipdepth/comparison`.
The release comparison uses CoreML FP32; the historical FP16 rejection is retained
in the local reports and the numerical results above.

Outputs, downloaded images and datasets remain outside git. Source rows metadata
and checksum are retained in `/private/tmp/zipdepth/nyu_probe/source.json`.

Published via Git LFS at commit `8c580f5`, with release tag `v0.11.0`.
All three release artifact URLs returned HTTP 200.

Publication target: https://huggingface.co/software-mansion/react-native-executorch-zipdepth,
release `v0.11.0`, `base/{backend}` layout.

## Vulkan grouped-convolution correction and Android verification

Device testing found that the initial Vulkan artifact ran but returned incorrect
values (range-normalized RMSE 132%, correlation 0.21 on cats). FP32 reproduced
the problem. Stage probes isolated grouped pointwise convolution: the 1.4.1
Vulkan pointwise shader reads from the first input group for all output groups.
The Vulkan-only adapter now expands non-depthwise grouped convolutions into
equivalent block-diagonal dense weights. Upstream source and checkpoint remain
unchanged. Seven adapted float references pass (worst normalized RMSE 0.00013%).

Corrected FP16 artifact: 12,796,162 bytes. On Samsung SM-G986B (Galaxy S20+),
all seven photos/illustrations passed, three executions each: all finite, worst
range-normalized RMSE 0.3925%, minimum correlation 0.999817. Warm execution was
approximately 69–108 ms. This supersedes the earlier offline-only Vulkan results.
Reports: `/private/tmp/zipdepth/vulkan_{adaptation,android_fixed}_verification.json`.
The release tag is refreshed with the corrected artifact; cached copies of the
original Vulkan file must be deleted before re-downloading.

## CoreML FP32 compute-unit comparison (2026-10-20)

Re-exported all four policies using the same model, CoreML precision FLOAT32,
iOS18 minimum deployment target, ExecuTorch 1.4.1. Four CPU threads; five warmups
then 30 interleaved timed runs per policy per image, three images (90 samples per
policy). Seven-image eager parity checks passed for every policy.

| Compute units | Median (ms) | P90 (ms) |
|---|---:|---:|
| ALL | 15.87 | 24.56 |
| CPU_ONLY | 33.18 | 80.23 |
| CPU_AND_GPU | 15.55 | 26.01 |
| CPU_AND_NE | 33.23 | 56.61 |

These are fresh local Mac measurements, not the earlier single-image 5.71 ms
measurement. Tail latency was variable, so compare policies within this run.
CPU_AND_NE showed no speedup over CPU_ONLY. Compute-unit policies permit hardware;
they do not prove which processor executed individual operations. No hardware
utilization tracing was performed. The published ALL preset remains unchanged.

Reproduce from the exporter root:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 exporter_models/zipdepth/.venv/bin/python exporter_models/zipdepth/compare_coreml_units.py
```

Artifacts and full timings/parity: `/private/tmp/zipdepth/coreml_units/`,
`comparison.json`. The export accepts an optional `compute_unit` config setting;
omitting it keeps ALL.

## Vulkan cross-scale pooling cleanup (2026-10-20)

The 384 input gives 24x24 and 12x12 cross-scale grids. Replace adaptive average
pooling in that path with exactly equivalent 2x2 stride-2 average pooling,
asserting divisibility at export. The Vulkan delegate supports regular pooling,
so the graph now has one delegate call and no portable CPU calls (previously
two delegates separated by adaptive pooling on CPU). Existing grouped-convolution
workaround is retained; no vendored source or checkpoint changes.

Seven adapted eager comparisons pass, worst normalized RMSE 0.000129%.
Offline float IO/texture validation passes; maximum extents [576,576,96].
Candidate: `/private/tmp/depth-optimized/zipdepth/vulkan/zipdepth_base_384_vulkan_fp16.pte`,
12,794,626 bytes, SHA256
`d4b675fbf7ea35c88065338d793f8af6bea4912a6acdda45346e21e5ca84078a`.

Validated on the reconnected Samsung SM-G986B (Galaxy S20+), Android 13.
The baseline and candidate were loaded in the same C++ process and alternated
per round, with three warmups and 15 timed runs for each of seven images
(105 timed samples per variant). All float32 outputs are finite, correctly
shaped, stable across repeated executions, and bit-identical to the baseline.
Worst normalized RMSE versus the float reference remains 0.3925%, minimum
correlation 0.999817.

Paired wall-clock timings (includes runtime orchestration and synchronization):

| Variant | Median | P90 |
|---|---:|---:|
| Published baseline (2 delegates / 1 CPU op) | 63.12 ms | 67.24 ms |
| Optimized (1 delegate / 0 CPU ops) | 54.51 ms | 58.59 ms |

Speedup: 1.158x, or 13.6% lower median latency. Counts are static delegate
boundaries, not shader dispatch counts; timings are whole-forward, not per-op.

`vulkan/benchmark_android.cpp` accepts baseline/candidate model paths, input
float file, side, output prefix, and timed-run count. Build with the Android NDK
against the current app ExecuTorch headers/shared libraries; run with both
Vulkan and runtime shared libraries in LD_LIBRARY_PATH. Its logs and paired
outputs can be analyzed using `compare_vulkan_android.py --help`.

Full device results: `/private/tmp/depth-optimized/zipdepth/android_verification.json`.
Raw logs and outputs: `/private/tmp/depth-optimized/zipdepth/zip_optimized/`.

Published the device-validated pooling optimization via Git LFS at `09c14a8`,
main and `v0.11.0`. Release SHA256 matches the artifact tested on the phone.
Delete cached prior Vulkan files before re-downloading the replacement.
