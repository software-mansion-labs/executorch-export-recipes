# Depth Anything V2 Small

Initial relative-depth export. The checkpoint is Apache-2.0 licensed and pinned
to `depth-anything/Depth-Anything-V2-Small-hf` revision
`5426e4f0f36572d16453bbda7a8389317b1bef99` in `_model.py`.
The `model.json` publication URL is a proposed destination; no artifacts have
been uploaded.

## Contract and preprocessing

- `forward`: float32 `[1, 3, 518, 518]` RGB -> float32 `[1, 518, 518]`.
- Normalize each RGB channel as `(pixel / 255 - mean) / std`, with means
  `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.
- Preserve raw relative-depth values. There is no conversion to meters,
  min/max normalization, grayscale conversion, or output quantization.
- This export has a fixed square input. The verification script deliberately
  stretches images to that square. Upstream's default processor preserves aspect
  ratio with variable dimensions divisible by 14; that is a different policy.
  The parity checks establish export correctness for identical input tensors,
  not equivalent depth quality between the two preprocessing policies.
- Image resizing, source-size output restoration, and the React Native task
  remain client-side work. The model itself returns its native 518x518 map.

## Reproduce

From the repository root:

```sh
uv sync --locked --project exporter_models/depth_anything_v2
uv run --project exporter_models/depth_anything_v2 exporter export depth_anything_v2 --size small --backend xnnpack --output-dir /private/tmp/depth_anything_v2/xnnpack
uv run --project exporter_models/depth_anything_v2 exporter export depth_anything_v2 --size small --backend coreml --output-dir /private/tmp/depth_anything_v2/coreml
uv run --project exporter_models/depth_anything_v2 exporter export depth_anything_v2 --size small --backend vulkan --output-dir /private/tmp/depth_anything_v2/vulkan

uv run --project exporter_models/depth_anything_v2 python exporter_models/depth_anything_v2/verify.py --backend xnnpack --artifacts-dir /private/tmp/depth_anything_v2 --image /path/to/photo.jpg
uv run --project exporter_models/depth_anything_v2 python exporter_models/depth_anything_v2/verify.py --backend coreml --artifacts-dir /private/tmp/depth_anything_v2 --image /path/to/photo.jpg
uv run --project exporter_models/depth_anything_v2 python exporter_models/depth_anything_v2/verify.py --backend vulkan --artifacts-dir /private/tmp/depth_anything_v2 --inspect-only
```

The verification script checks serialized I/O against the local spec, backend
IDs, fallback instructions, and numerical parity. Runtime inputs are contiguous.
It writes `<backend>_verification.json` beside the artifacts. The cases are the
supplied photograph, its 90-degree rotation, seeded random input, and zero input.
XNNPACK must pass `allclose(atol=1e-4, rtol=1e-4)`. The FP16 screening threshold
is RMSE below 1% of the reference range and correlation above 0.99. This is a
numerical export check, not an evaluation on a labeled depth dataset.

`--trace trace.etdp` requires a runtime built with event tracing. The installed
1.4.1 wheel exposes the tracing API but emits no data; the verifier reports an
error rather than silently accepting a missing trace.

## Results on 2026-10-19

Apple M1 Pro, macOS 26.5.2, four PyTorch and ExecuTorch threads. Warm latency is
the median of five forward calls after warmup; it excludes preprocessing, model
load, and CoreML compilation. The real input was the Transformers documentation
COCO cats sample, downloaded from
`https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png`.

| Backend | Bytes | Warm latency | Validation |
| --- | ---: | ---: | --- |
| XNNPACK FP32 | 99,023,364 | ~500 ms | All four cases pass; worst absolute error 0.00002599 |
| CoreML FP16 | 50,181,365 | ~57 ms | All four cases pass; worst RMSE/range 0.228%, minimum correlation 0.999964 |
| Vulkan FP16 | 49,595,524 | Unmeasured | Serialized contract and texture checks pass; runtime untested |

XNNPACK has 50 delegated partitions and 79 portable kernel calls. The remaining
operator kinds are `native_layer_norm`, `expand_copy`, dim-order copy, and
`squeeze_copy`. CoreML and Vulkan each contain one delegate call and no portable
operators. These are serialized graph counts, not runtime profiling ratios.
Per-operation timing is unavailable without an event-tracing runtime.

Vulkan uses `force_fp16=True` and a 2048x2048x2048 texture limit. Inspection of
the serialized graph finds 647 3D textures and two buffers, with maximum texture
extents `[1369, 1370, 1370]`, below that limit. Export uses released ExecuTorch
1.4.1; compatibility with the application's fork runtime still needs a device
run. No Android device was connected during this work. CoreML was exercised on
macOS, not an iPhone; device placement and numerical behavior need a separate
check before publishing.

Artifacts and JSON reports from this run are in `/private/tmp/depth_anything_v2`.
Weights, environment files, and compiled artifacts are not committed.

## XNNPACK INT8 experiment

The optional `int8` variant uses per-channel INT8 weights and dynamic INT8
activations for `nn.Linear` layers. Convolutions and other operations retain
floating point execution. A single dummy forward initializes weight observers;
dynamic activation ranges are determined at inference, so this recipe does not
need image calibration. Public input and output tensors stay float32. FP32
remains the default.

```sh
uv run --project exporter_models/depth_anything_v2 exporter export depth_anything_v2 --size small --backend xnnpack --variant int8 --output-dir /private/tmp/depth_anything_v2/xnnpack
uv run --project exporter_models/depth_anything_v2 python exporter_models/depth_anything_v2/verify.py --backend xnnpack --variant int8 --artifacts-dir /private/tmp/depth_anything_v2 --image /private/tmp/depth_anything_v2/evaluation
```

For evaluation, download the four images named in `evaluation.json` from its
pinned `images_base_url` into the supplied directory. This directory must contain
only the intended evaluation JPGs. The verifier also checks seeded random and
zero tensors. `--variant fp32` runs the same cases on the baseline. Reports are
written separately as `xnnpack_int8_verification.json` and
`xnnpack_fp32_verification.json`.

| Variant | Bytes | Warm latency on demo17 | Size reduction |
| --- | ---: | ---: | ---: |
| FP32 | 99,023,364 | 502 ms | Baseline |
| Dynamic INT8 linears | 35,525,508 | 354 ms | 64.1% |

INT8 is about 1.42x faster in this host measurement. On the four held-out real
images, RMSE divided by the FP32 depth range is 0.47%-1.34%, and correlation is
0.999104-0.999970. The provisional INT8 screen is below 2% normalized RMSE and
above 0.99 correlation. All four images and the zero tensor pass; random input
fails the error screen (3.32% normalized RMSE, correlation 0.990703). The verifier
therefore exits nonzero and records `passed: false`. This candidate needs broader
image/depth evaluation before promotion; it is not equivalent to FP32.

The INT8 graph contains 62 XNNPACK delegate calls and 78 portable kernel calls.
The float artifact remains unchanged. A fully static INT8 recipe was also tried,
with per-channel weights and activation calibration on upstream demo01-demo16.
It produced a 24 MiB file but severely degraded held-out depth structure
(correlations between -0.214 and 0.418). Its validation process also aborted
during repeated inference after recording those cases. It is rejected and is
not registered as a supported variant. Its artifact and log remain under
`/private/tmp/depth_anything_v2/experiments` and `/private/tmp/depth_int8_verification.log`.


## XNNPACK attention optimization (2026-10-19)

Native macOS sampling of the existing INT8 export found 3,420 of 3,810 sampled
main-thread stacks inside XNNPACK invocation, 284 in portable `expand_copy`,
and 90 in portable layer normalization. These are stack-sampling shares, not
per-operator wall times; XNNPACK also executes on worker threads. The pip runtime
does not emit ETDump timing events, and the separate profiler repository was
not accessible without authentication, so no precise per-operator timings are
claimed.

The XNNPACK export now rewrites DINOv2 attention for its fixed batch of one:
heads become the batch dimension of explicit rank-3 `bmm` operations. This
avoids the expansion copies emitted by rank-4 broadcasted `matmul`. Scaling
moves from the large attention-score tensor to the much smaller query tensor.
Checkpoint weights, input/output shape, normalization and quantization remain
the same. The shared model builder and CoreML/Vulkan exports are unchanged.

On an M1 Pro, four threads, alternating ten warm calls per artifact at 518x518:

| Candidate | Baseline median | Candidate median | Result |
| --- | ---: | ---: | --- |
| Rank-3 attention | 363.95 ms | 326.26 ms | Identical on checked inputs |
| Rank-3 plus query scaling | 363.63 ms | 318.90 ms | Selected; identical on checked inputs |
| Rank-3 plus decomposed layer norm | 360.51 ms | 342.43 ms | Rejected; quantized predictions changed |

The selected graph has 38 delegate calls and 30 portable calls, down from 62
and 78 respectively. Layer normalization stays on the existing optimized native
kernel. The artifact is 35,481,220 bytes. The speedup is about 1.14x, or 12.3%
less model execution time. It is a host measurement and excludes image
preprocessing, output resizing, worklet transfer, and simulator contention.
The INT8 quality limitations described above still apply.

The original artifact is preserved at
`/private/tmp/depth_anything_v2/xnnpack/depth_anything_v2_small_xnnpack_int8.pte`.
The optimized export is in
`/private/tmp/depth_anything_v2/optimized/depth_anything_v2_small_xnnpack_int8.pte`.
Sampling data and candidate comparisons are under
`/private/tmp/depth_anything_v2/optimization/` and
`/private/tmp/depth_xnnpack_sample.txt`.

## Apply rank-three attention to the published FP32 export (2026-10-20)

The XNNPACK export function already applies the INT8 attention rewrite to float
models too. Re-exported FP32 with that function: delegate/portable calls drop
from 50/79 to 26/31, and expands drop from 49 to 1. Outputs are bit-identical
on seven saved image inputs, a zero tensor, and a seeded random tensor.

Four-thread M1 Pro paired benchmark: median 509.56 ms to 465.58 ms (1.09x),
three images with two warmups and five alternating baseline/candidate runs each.
Artifact: `/private/tmp/depth-optimized/v2/xnnpack/depth_anything_v2_small_xnnpack_fp32.pte`.
Full results: `/private/tmp/depth-optimized/comparison.json`. Reproduce using
`tools/compare_depth_xnnpack.py`; INT8, CoreML and Vulkan were not changed.

Published the verified XNNPACK update via Git LFS at `5ded482`, main and
`v0.11.0`. Delete cached prior XNNPACK files before re-downloading.
