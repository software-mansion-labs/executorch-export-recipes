# migan notes

- **Layout:** `{backend}/{size}/config.json`, both locally (`xnnpack/256/`,
  `xnnpack/512/`) and in the HF repo (`xnnpack/256/migan_256_xnnpack_fp16.pte`),
  backend on top as in the spec's model-card example and kokoro (other
  multi-size repos differ). The HF repo holds the root `config.json`,
  `README.md`, `LICENSE` and one `<backend>/<size>/` dir per export.

Tracks react-native-executorch issue #1373.

- **XNNPACK ships fp16 only, no fp32 and no int8.** fp16 equals fp32 to
  ~1e-3 and is clearly faster on an ARM phone. int8 was exported and measured and is
  **buggy on the RNE Android runtime**: the same `.pte` that is fine on the
  host runtime returns a saturated, wrong image on the S23 and on the Android
  emulator (mean-in-hole 0.80 / 1.00 instead of 0.49), on top of visible
  speckles already on the host (25 dB vs fp32). Root cause: the XNNPACK
  weights cache (on in the Android/iOS builds) hands nodes with same-named
  quantized kernels each other's packed weights; fixed in the ET fork on
  2026-10-15, not yet in an RNE release. Details in "XNNPACK precision
  study" below; `xnnpack/export.py` refuses `quantized` variants with a
  pointer here.

- **Source:** https://github.com/Picsart-AI-Research/MI-GAN (MIT code, MIT
  weights), pinned to `SOURCE_COMMIT` in `_migan.py`. The vendored file is
  upstream's `lib/model_zoo/migan_inference.py`, verbatim except for one
  marked `EXPORT PATCH`: the activation gain is `float(np.sqrt(2))` instead
  of the numpy float64 upstream keeps, because `torch.export` traces a numpy
  scalar as a tensor and `if act_gain != 1:` then becomes data-dependent
  control flow. The in-place ops upstream uses (`leaky_relu(inplace=True)`,
  `add_`) export fine and are left as-is.
- **Weights** are fetched at export time into `weights/` (gitignored) and
  sha256-checked against `WEIGHTS_SHA256`. Upstream publishes them only in a
  public Google Drive folder; `drive.usercontent.google.com/download?...&confirm=t`
  returns the bytes with plain `urllib` (the `drive.google.com/uc` link answers
  with an HTML virus-scan interstitial for files over ~25 MB). The HF repo holds
  only exported artifacts, like every other SWM model repo; mirror the `.pt`
  files there only if Drive ever breaks.
- **Sizes:** `256` and `512` are the Places2 (scenery) checkpoints at that
  resolution; `faces_256` is the FFHQ (faces) checkpoint, same architecture
  as `256`, different weights, upstream ships it only at 256. The size token
  is a checkpoint id in the model.json sense (like kokoro's `german`), so the
  exporters map it through `RESOLUTION_FOR_SIZE` / `CHECKPOINT_FOR_SIZE` in
  `_migan.py` instead of `int(size)`. Files: `migan_faces_256_<backend>_fp16.pte`.
- **faces_256 verified (2026-10-15):** XNNPACK fp16 `.pte` vs fp32 eager on
  upstream's `ffhq_256_freeform` examples, inside the hole: 52 to 62 dB
  (mean 56). On the same faces the Places2 `256` checkpoint differs by
  0.05 to 0.09 mean abs and visibly smears eyes and mouths where the face
  checkpoint reconstructs them, which is the reason to ship it. XNNPACK smoke-tested on the Android emulator
  (loads, `XnnpackBackend`, sensible output); CoreML and Vulkan only
  statically checked (one delegate call, zero portable ops each), not yet run
  on a phone.
- **Graph I/O** (`InpaintingWrapper`): `image [1,3,H,W] f32` RGB in `[0,1]`,
  `mask [1,1,H,W] f32` with **1 = known pixel, 0 = hole**; output
  `[1,3,H,W] f32` in `[0,1]`, the whole frame with the hole filled. Upstream's
  normalisation (`image*2-1`, `mask-0.5`, 4-channel cat, `out*0.5+0.5`, clamp)
  is inside the graph. **Mask polarity trap:** upstream's example masks are
  white = hole, so they must be inverted before feeding; the README's "white
  denotes the known region" describes its own `read_mask` convention, not the
  PNGs. Cropping to the hole's bounding box and blending the result back with
  a dilated + blurred mask are NOT in the graph (data-dependent indexing) —
  that is the app-side task, a port of `MIGAN_Pipeline.forward` in upstream's
  `scripts/create_onnx_pipeline.py`.
- **XNNPACK fp16** (`xnnpack/export.py`, the only XNNPACK variant shipped):
  `model.half()` behind an fp32-I/O wrapper, so the app contract stays fp32.
  24.2 MB (256) / 25.9 MB (512).
- **Verified** (ET 1.4.1 runtime on host, torch 2.14.0): the fp32 build
  (2026-10-13, since dropped) matched eager on upstream example images to max
  abs 3e-6 at both sizes, with a clean visual check on
  `places2_512_object/1.png` (sheep removed); the shipped fp16 build
  (2026-10-14) matches that fp32 output inside the hole to 55 dB / mean abs
  diff ~1e-3 (see the precision study below), and the harness checks
  `method_meta` against the config on every export.
- **Smoke-tested on a Samsung Galaxy S23** through a local, uncommitted screen
  in the react-native-executorch computer-vision example app (low-level
  `loadModel` + `execute` on a synthetic gradient with a square hole): the fp32
  build on 2026-10-13, the shipped fp16 files on 2026-10-14 and again on the
  Android emulator on 2026-10-15 after the final export. Both sizes load,
  report `XnnpackBackend` for `forward`, and return the host result (fp16
  matches fp32 to 4 decimals). This is not the RNE inpainting task (it does
  not exist yet), so no timings are recorded here; 512 was noticeably less
  than 4x the cost of 256, a fixed cost the profiler run below pins on the
  partition split around the portable `upsample_nearest2d` ops.
- **Delegation** (PyPI ExecuTorch 1.4.1, the env this model pins): everything
  lands on XNNPACK except `upsample_nearest2d` (12 ops at 256, 14 at 512),
  which that partitioner does not take, so the graph is 7 / 8 partitions. The
  upstream `Upsample2d` is nearest-x2 followed by a `[[1,0],[0,0]]`
  checkerboard multiply, i.e. zero insertion. Fixed in the ET fork, see
  "Profiling" below; with the fork the export also swaps `Upsample2d` for a
  depthwise transposed conv, see "Export-time graph rewrites". The export folds constants before partitioning
  (`_fold_constants` in `xnnpack/export.py`), so the only portable casts left
  are the three fp32 I/O casts of the wrapper; before that the fp16 build
  carried one zero-dim fp32 to fp16 cast per scalar constant (65 / 73).
- **Testing trap:** when checking a `.pte` from a PIL/numpy image, `permute`
  leaves the tensor non-contiguous and the ET runtime reads the raw buffer as
  contiguous NCHW — call `.contiguous()` first or the output is garbage (and
  the "parity failure" looks identical for every build you try).
- Env pins stable `torch==2.14.0` from PyPI rather than a dated nightly (the
  nightly index prunes old builds; facemesh does the same).

## Export-time graph rewrites (2026-10-20)

Two numerically identical rewrites of the vendored model, applied after
`load_state_dict` in the export scripts (both live in `_migan.py` next to
`vulkanize`), after the per-op profiles of all three backends showed the
remaining time outside the convolutions sitting in `lrelu_agc` and in the
upsampling filter:

- **`fold_activation_gain()`** (all three backends). Upstream's `lrelu_agc`
  is `clamp(sqrt(2) * leaky_relu(y), -256, 256)`: three elementwise passes
  after every conv, 26% of the forward on XNNPACK and Vulkan, ~54% of the
  CoreML op time. `leaky_relu` is positively homogeneous and monotonic, so
  `clamp(g * lrelu(y), -c, c) == lrelu(hardtanh(g * y, -c / alpha, c))`, and
  every `y` is a conv output (optionally through the nearest upsample and
  the constant noise add, both linear): the gain goes into the conv weight
  and bias (and `noise_strength`), the clamp moves in front of the
  `leaky_relu` with bounds `(-1280, 256)`, and is written as `hardtanh`
  because that is what the conv-activation fusion passes of XNNPACK
  (`FuseActivationPass`) and Vulkan (`FuseClampPass`) match; `aten.clamp`
  would stay a separate op. fp32 eager, folded vs original: max abs diff
  3e-5 / 1e-5 / 1e-5 (256 / 512 / faces_256); the fp16 error vs fp32 is
  unchanged (1.3e-2 / 2.2e-2 / 8e-3 vs 1.1e-2 / 2.6e-2 / 5.5e-3 before).
  Two traps: coremltools rejects an `int` clip bound against an fp tensor
  (`beta must have the same data type as x`), so the bounds are floats; and
  the Vulkan graph builder keyed its constant-scalar cache by Python value,
  so `256.0` collided with `groups=256` and the runtime failed at load with
  `Expected value to have type Int, got DOUBLE` (fixed in the fork, branch
  `vulkan-scalar-cache-float-int`, commit af3a5336c; export-side only).
- **`upsample_as_deconv()`** (XNNPACK with the fork, CoreML). Upstream's
  `Upsample2d` is nearest x2, the checkerboard multiply (zero insertion),
  `F.pad(2, 1, 2, 1)` and a 4x4 depthwise FIR conv at full resolution. Zero
  insertion followed by a FIR filter is a stride-2 transposed conv:
  `conv_transpose2d(x, w.flip(-1, -2), stride=2, padding=1, groups=C)`
  matches to 5e-7 for every (channels, resolution) in the model, runs on the
  low-resolution input (a quarter of the filter MACs) and removes the
  upsample, mask and pad passes. Backend support decides where it is used:
  - XNNPACK runs a transposed conv through its per-group subconvolution
    IGEMM; with one channel per group that was 88% of the forward, ~8x the
    whole model at 256 on the host. Fixed in the fork
    (commit 47a301242 on `xnnpack-upsample-nearest2d`):
    `DecomposeDepthwiseConvTransposePass` lowers a depthwise transposed conv
    that is a clean x s upscale into s*s stride-1 depthwise convs on the
    low-resolution input (polyphase: phase (ay, ax) correlates with every
    s-th tap, through an asymmetric zero pad that `FusePadIntoConvPass`
    folds), a channel concat and XNNPACK's block-major `depth_to_space_2d`
    (new `et_xnnpack` custom op + visitor; `aten.pixel_shuffle` is
    channel-major and does not map). `xnnpack/export.py` applies the rewrite
    only when that pass is importable, so a stock PyPI ExecuTorch still
    exports the upstream block (fold only, 7 / 8 partitions as before)
    instead of a correct but ~8x slower file.
  - Vulkan keeps the upstream block: the fork's generic `conv_transpose2d`
    shader took 52% of GPU time for the twelve filters, 1.6x the shader
    time of the whole model at 256, far worse than nearest + mul + `conv2d_dw`.
  - CoreML takes the transposed conv natively (`conv_transpose`, 0.4% of op
    time, on the ANE): 231 ops instead of 324 at 256.

## Backends beyond XNNPACK (2026-10-13)

- **iOS: CoreML, not MLX.** The repo's own measurements rule MLX out for conv
  CV: `docs/ios-backend-and-quantization-findings.md` has MLX 19x slower than
  CoreML on efficientnet and `skills/backends/mlx` records every CV MLX export
  as slower than its CoreML twin and "DON'T SHIP". MI-GAN is a plain conv
  encoder-decoder, so no `mlx/` dir was made.
- **CoreML fp16** (`coreml/export.py`, same build as xnnpack, inline
  `CoreMLPartitioner`, iOS18 target): 13.0 MB (256) / 14.8 MB (512). Static
  check of the `.pte` (deserialised program): one `CoreMLBackend` delegate call,
  zero portable ops, I/O matches the config. `compute_unit` is left at the
  default `ALL`, swept on device, keep it (below).
- **Smoke-tested on an iPhone 16 / iOS 26.6 (2026-10-13)** with the same local
  screen: both sizes load, report `CoreMLBackend`, and match the fp32 host
  reference to ~1e-4 (fp16-level). A compute-unit sweep (`ALL`, `CPU_AND_NE`,
  `CPU_AND_GPU`, `CPU_ONLY`) showed `ALL` landing on the Neural Engine
  (identical output and speed to `CPU_AND_NE`) and being the fastest by a wide
  margin, far ahead of XNNPACK on the same phone -- the opposite of
  style_transfer's ANE pathology, which is why the sweep is per model. No
  compute_unit pin needed. First CoreML load compiles the model (seconds, once
  per install).
- **Vulkan fp16** (`vulkan/export.py`, `force_fp16`, own `vulkan/pyproject.toml`
  pinned to the software-mansion-labs executorch fork like every other Vulkan
  export): 12.4 MB (256) / 14.1 MB (512). Static check: one `VulkanBackend`
  delegate call, zero portable ops -- Vulkan takes `upsample_nearest2d`, which
  XNNPACK leaves on CPU. **`uv sync` of the fork pin cannot build on a fresh
  machine**: upstream ExecuTorch's CMake refuses any checkout directory not
  named exactly `executorch` (pytorch/executorch#6475) and uv names git
  checkouts by commit hash. Like the MLX models (`skills/model-conventions`),
  the pyproject declares the pin and the export is run in a source-built dev
  env: these files came from a source-built checkout of the fork (editable
  executorch 1.4.1+f8b41ab = the fork's `@ms/separate-backends-1.4.1` lineage
  plus the Vulkan mean/safe_idx fix, torch 2.13.0, python 3.13) via
  `PYTHONPATH=src python -c 'from exporter.cli import main; main()' export migan
  --backend vulkan ...`.
- **Vulkan needs one graph rewrite** (`Downsample2dVulkan` / `vulkanize()` in
  `_migan.py`, applied only by `vulkan/export.py`): the first export loaded on
  the Galaxy S23 with `Could not find ShaderInfo with name conv2d_dw_sned_half`.
  The runtime routes a depthwise conv whose stride != dilation to the generic
  `conv2d_dw_sned` shader, and the prebuilt RNE Vulkan lib (nativeLibsVersion
  0.10.4) ships only its tiled 3x3/5x5 variants; upstream's `Downsample2d` is a
  4x4 stride-2 depthwise blur, which fits neither. Running that conv at stride 1
  (plain `conv2d_dw`, shipped) followed by `avg_pool2d(kernel=1, stride=2)`
  (shipped) is the same arithmetic: eager original vs rewritten max abs diff
  8e-6. 6 blocks swapped at 256, 7 at 512. XNNPACK / CoreML keep the upstream
  block.
- **Smoke-tested on the Galaxy S23 (Adreno, 2026-10-13)** with the same local
  screen: both sizes load (`VulkanBackend`) and return the same result as
  XNNPACK. Forward was several times faster than XNNPACK fp32 on the same
  phone and still clearly ahead of the fp16 build now shipped; only the load
  and the first call take longer, for the shader compile. Vulkan is the
  Android default candidate for this model.
- **Download cache trap when re-testing on device:** the RNE fetcher keys its
  cache on the URL path (a `?v=2` query does not bust it), so a re-exported
  file at the same URL is never re-downloaded. Delete
  `/storage/emulated/0/Android/data/<app>/files/react-native-executorch/*` via
  adb, or rename the file.

## Profiling (2026-10-15, executorch-profiler)

Run with the sibling [executorch-profiler](https://gl.swmansion.com/swm-ai/executorch-profiler)
against an event-traced ExecuTorch built from the SWM fork at
`@ms/separate-backends-1.4.1` (the branch the RNE runtime is built from) with
`EXECUTORCH_ENABLE_EVENT_TRACER=ON`, `EXECUTORCH_BUILD_COREML=ON`,
`EXECUTORCH_BUILD_MLX=OFF` (MLX needs the Metal toolchain and is not needed
here). Inputs are real upstream example pairs, 10 forwards each. Shares
below are of `Method::execute` time on this Mac; absolute numbers are
host-side and not recorded.

**XNNPACK fp16, where the time goes:**

| | 256 | 512 | faces_256 |
| --- | --- | --- | --- |
| delegated (XNNPACK) | 87% | 82% | 87% |
| portable `upsample_nearest2d` (12 / 14 / 12 calls) | 11% | 14% | 12% |
| portable `_to_dim_order_copy` (65 / 73 / 65 calls) | 2% | 4% | 2% |
| inside the delegate: `Transpose` (NCHW<->NHWC) | 22% | 25% | 22% |
| inside the delegate: GEMM (1x1 convs) | 28% | 21% | 27% |
| inside the delegate: depthwise conv | 10% | 10% | 10% |
| inside the delegate: pad / mul / lrelu / clamp / add | 23% | 21% | 24% |

- **Delegation is complete apart from `upsample_nearest2d`**, as the static
  check said, and every delegated op reports 100%.
- **The partition split is the real cost, not the upsample itself.** The
  portable upsample ops cut the graph into 7 (256) / 8 (512) XNNPACK
  partitions, and each partition converts its inputs and outputs between
  NCHW and NHWC: the `Transpose` ops are a quarter of the forward, on top of
  the 11 to 14% spent in the upsample kernels. Roughly 35 to 40% of XNNPACK
  time is partitioning overhead. That is the fixed cost suspected from the
  256 vs 512 smoke test.
- **Fixed in the ET fork (2026-10-19, branch `xnnpack-upsample-nearest2d`,
  commits 223d80999 + 59b1b7c6c, not yet merged):** `upsample_nearest2d` x2
  is lowered to XNNPACK as `concatenate4` + `depth_to_space_2d` (XNNPACK has
  no nearest resize node and its `static_broadcast` is not implemented at
  run time); the channels-last pass no longer lets a scalar operand
  (`x * gain` in `lrelu_agc`) or a `[H, W]` operand (the noise map) drag the
  activation back to NCHW; the zero `F.pad(2, 1, 2, 1)` before each 4x4 blur
  filter is folded into that convolution's (asymmetric) padding, and the
  fp32/fp16 I/O casts are lowered as XNNPACK `Convert` nodes. Inspecting the pass showed the transposes were mostly caused by
  those scalars, not by the partition boundaries: 128 layout conversions per
  forward at 256, of which 90 sat around the 45 scalar multiplies and only
  24 at the upsample boundaries. A depthwise `conv_transpose2d(stride=2)`
  rewrite of `Upsample2d`, tried earlier, gave one partition but was slower
  on the S23. Re-profiled with the patched fork on the same inputs:

| | 256 | 512 | faces_256 |
| --- | --- | --- | --- |
| delegated (XNNPACK) | 100% | 100% | 100% |
| XNNPACK partitions per forward / portable ops | 1 / 0 | 1 / 0 | 1 / 0 |
| GEMM (1x1 convs) | 47% | 41% | 47% |
| depthwise conv (blur + 3x3), pad folded in | 16% | 19% | 16% |
| `lrelu_agc` as Multiply + LeakyReLU + Clamp | 26% | 27% | 26% |
| nearest upsample (`Copy` + `Depth To Space`) | 6% | 7% | 6% |
| `Convert` (3 I/O casts), `Transpose`, `Static Reshape` | <1% | <1% | <1% |
| `Constant Pad` | 0 | 0 | 0 |
| `Method::execute` vs the shipped build (host) | -45% | -51% | -45% |

  Output is unchanged (max abs diff 2e-3 vs the shipped files, fp16 noise,
  same error vs fp32 eager). Before the pad fold and the casts, with only
  the first commit, the forward was -29 / -28 / -31% with 97.6 / 94.5 /
  97.5% delegated: the NHWC `Constant Pad` alone had grown to 13 / 22 / 14%.
  The `.pte` files on HF are still the PyPI-ET build: the single-partition
  graph needs the fork's partitioner at export time and the new
  `XNNDepthToSpace2d` node in the runtime, so it ships once the branch is
  merged and the RNE native libs are rebuilt. What is left is the model's
  own arithmetic; the only structural item is `lrelu_agc` running as three
  elementwise passes (kernel fusion in ET, or folding the gain into the next
  conv at export).
- **GEMM, not 3x3 conv**: MI-GAN's `SeparableConv2d` is depthwise 3x3 +
  pointwise 1x1, so the heavy op is the 1x1 GEMM; depthwise convs are 10%.
- **Re-profiled after the export-time rewrites (2026-10-20)**, fork with
  commit 47a301242, same inputs, 10 forwards. XNNPACK `Method::execute` on
  the host: -16 / -18 / -15% vs the fork build above and -54 / -60 / -53%
  vs the shipped PyPI build. Still one partition, no portable ops. Shares of the delegate's op time:

| | 256 | 512 | faces_256 |
| --- | --- | --- | --- |
| GEMM (1x1 convs) | 56% | 50% | 56% |
| depthwise conv (3x3, down blur, 4 polyphase filters per upsample) | 19% | 20% | 18% |
| `LeakyReLU` | 10% | 11% | 10% |
| `Copy` (channel concat of the 4 phases) + `Depth To Space` | 7% | 9% | 7% |
| `Add` (noise, skips, RGB) | 4% | 5% | 4% |
| `Clamp` (12 hardtanh after the noise add + the output clamp) | 2% | 3% | 3% |
| `Multiply` (3 wrapper muls), `Convert`, `Transpose`, `Reshape` | <1% each | <1% | <1% |

  The 57 gain multiplies are gone; 45 of the 57 clamps fused into their
  conv (`output_min/max`), the 12 after the noise add fuse into XNNPACK's
  `add` only nominally (XNNPACK re-emits a `Clamp` node for a clamped
  fp16 binary op). What is left is the model's arithmetic plus `LeakyReLU`,
  which XNNPACK has no fused form for.

**CoreML fp16, compute plan** ([`coreml_compute_plan.py`](../../tools/executorch-profiler/coreml_compute_plan.py), compute units `ALL`
as compiled; what the CoreML runtime on this Mac would choose, iOS may differ):

| | 256 | 512 | faces_256 |
| --- | --- | --- | --- |
| ops on ANE | 274 / 324 (85%) | 283 / 370 (77%) | 274 / 324 (85%) |
| ops on GPU | 50 (15%) | 87 (23%) | 50 (15%) |
| ops on CPU | 0 | 0 | 0 |
| after the rewrites (2026-10-20): ops on ANE | 227 / 231 (98%) | 246 / 263 (94%) | 227 / 231 (98%) |
| after the rewrites: ops on GPU | 4 | 17 | 4 |

- The split follows resolution: at 256 the first 272 ops run on the ANE and
  the last 50, the full-resolution decoder block plus the RGB output, on the
  GPU; at 512 the first full-resolution encoder block goes to the GPU as
  well and the middle of the network stays on the ANE. Nothing on the CPU.
  Matches the on-device sweep, where `ALL` and `CPU_AND_NE` were
  indistinguishable: the ANE carries the bulk, the GPU takes the
  full-resolution ends.

**Vulkan fp16 and CoreML fp16, per op (2026-10-19)**: the profiler's ET
build was rebuilt with `EXECUTORCH_BUILD_VULKAN=ON` and runs on MoltenVK
(`DYLD_LIBRARY_PATH=/opt/homebrew/lib`); with the event tracer on, the
fork's `VulkanBackend` enables its query pool and logs one row per shader
dispatch, and the CoreML delegate reports its ops the same way. Both are
single-delegate graphs with no portable ops at any size. Shares of the
per-op sum (Apple GPU / ANE, so shares only):

| | Vulkan 256 / 512 / faces_256 | CoreML 256 / 512 / faces_256 |
| --- | --- | --- |
| 1x1 convs (`conv2d_pw_tiled` / `conv`) | 55 / 43 / 54% | 34 / 34 / 34% |
| depthwise convs | 12 / 14 / 12% | inside `conv` |
| `lrelu_agc` as three passes (leaky_relu, mul, clamp) | 26 / 33 / 26% | 54 / 53 / 54% |
| nearest upsample | 1 / 1 / 1% | 2 / 2 / 2% |

Upsample is a non-issue on both. After the convolutions the next item was
`lrelu_agc` running as three elementwise passes; see "Export-time graph
rewrites" for the fold. Re-profiled with it (2026-10-20), shares of the
per-op sum:

| | Vulkan 256 / 512 / faces_256 | CoreML 256 / 512 / faces_256 |
| --- | --- | --- |
| 1x1 convs (`conv2d_pw_tiled`, 47% of them with the clamp fused) | 66 / 54 / 65% | 33 / 31 / 33% (`conv`) |
| depthwise convs (`conv2d_dw`, `..._3x3_clamp`) | 14 / 18 / 14% | inside `conv` |
| `leaky_relu` | 9 / 12 / 9% | 26 / 27 / 26% |
| `hardtanh` left standalone (after the noise add) | 2 / 3 / 2% | 26 / 28 / 26% (`clip`, none fused) |
| `mul` (mask + wrapper) | 2 / 2 / 2% | <1% |
| `add` | 4 / 6 / 4% | 15 / 13 / 15% |
| nearest upsample + pad | 3 / 4 / 3% | `conv_transpose` <1% |

Vulkan shader time on the host dropped 18 / 22 / 18%. CoreML is
dispatch-bound: the op count fell from 324 to 231 at 256 and the ANE share
rose to 98%, but `clip` and `leaky_relu` stay separate ops and the host
op-sum moved only ~3%; the on-device check is what counts there.

## XNNPACK precision study: fp32 vs fp16 vs int8 (2026-10-14)

To decide what XNNPACK should ship, fp16 and static int8 (PT2E, per-channel
weights, `XNNPACKQuantizer`) were exported alongside fp32 and measured on
upstream's example sets. int8 is calibrated on the free-form sets
(`places2_512_freeform` for 512; that plus `places2_256_freeform` for 256) and
evaluated on sets it never saw: `places2_512_object` and `ffhq_256_freeform`
(faces, out of domain, random masks). Metrics are computed **inside the hole
only**, against the fp32 export's output and against the original pixels;
"PSNR vs GT" is only meaningful on random-mask sets (on the object set the
original still contains the removed object). Measured on this Mac.

| 512 | MB | PSNR vs fp32 | SSIM vs fp32 | mean abs diff | PSNR vs GT (ffhq) |
| --- | --- | --- | --- | --- | --- |
| fp32 | 28.0 | - | 1.0000 | 0 | 17.64 |
| fp16 | 25.9 | 55 dB | 0.9999 | 0.001 | 17.65 |
| int8 | 7.3 | 25.4 dB (min 21) | 0.937 | 0.034 | 17.10 |
| int8, RGB path kept fp32 | 8.8 | 24.1 dB | 0.927 | 0.040 | 16.99 |

| 256 | MB | PSNR vs fp32 | SSIM vs fp32 | mean abs diff | PSNR vs GT (ffhq) |
| --- | --- | --- | --- | --- | --- |
| fp32 | 24.7 | - | 1.0000 | 0 | 17.80 |
| fp16 | 24.2 | 56 dB | 0.9999 | 0.001 | 17.80 |
| int8 | 6.5 | 26.6 dB (min 21) | 0.949 | 0.030 | 17.55 |

- **fp16 is numerically fp32** (55 dB, mean abs diff ~1/1000) and only 8%
  smaller (the delegate keeps most weight bytes in fp32, as facemesh records),
  but on the S23 smoke-test screen it ran about twice as fast as fp32 (ARM
  fp16 NEON arithmetic; on the Mac the gain was much smaller) with output
  matching fp32 to 4 decimals. int8 ran faster still, but see below.
- **int8 on host is 3.8x smaller, with visible damage**:
  blue speckles in smooth regions (sky, haze) and along high-contrast edges;
  PSNR to fp32 inside the hole
  ~25 dB, i.e. ~8/255 average error, and 0.3-0.5 dB lost against ground truth.
  Keeping the RGB skip path (`fromrgb`, `torgb`, image upsample) in fp32 does
  NOT help, so the error is in the feature path -- plausibly the activation
  range (the `lrelu_agc` clamp is +/-256, so per-tensor 8-bit steps are coarse).
- **int8 on the RNE Android runtime computed the wrong thing, root cause
  found and fixed in the ET fork (2026-10-15).** The same `.pte` gave
  mean-in-hole 0.49 on the host runtime but 0.80 (256) / 1.00 saturated (512)
  on the S23 and, identically, on the Android emulator. The host build of the
  fork matched the host wheel, so it was build configuration: the fork's
  `android.cmake` / `apple_common.cmake` presets turn on
  `EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE`, the pybind preset does not, and
  `executor_runner` built on the Mac with the cache on reproduces the Android
  numbers bit for bit. The cache keys packed weights on the unpacked kernel's
  name (+ bias name). The exporter dedups constants by content hash, so
  MI-GAN's Downsample2d (blur) and Upsample2d (4 x blur) depthwise kernels,
  identical as int8 codes with different scales, share one name, and every
  512-channel block shares it again with different activation scales, which
  XNNPACK folds into qc8 packing. A `look_up` hit is final for XNNPACK (it
  never packs on a hit), so the second node ran on the first node's packing.
  fp16/fp32 are immune only because their kernel bytes differ. Fix: the cache
  name of a quantized kernel now carries its scale tensor name / zero point
  and the consuming node's input/output quantization; branch
  `xnnpack-weights-cache-quant-names` in the SWM fork, with a gtest. Verified:
  cached host runner bit-identical to uncached for int8 256/512, the
  single-partition variant and the minimal U-Net repro, and the patched
  `executor_runner` on the Android arm64 emulator matches the host. Not yet
  in an RNE native-libs release, so the shipped Android lib still has it.
  Upstream has the same cache design, PR candidate.
- **Decision (2026-10-15): XNNPACK ships fp16 only.** fp32 is dropped (same
  output, slower) and int8 is dropped (quality loss on host; the Android
  divergence is a runtime bug, fixed in the fork, see above). The int8 recipe
  (PT2E, `XNNPACKQuantizer`, per-channel weights, calibration on a sparse
  clone of upstream `examples/` at `SOURCE_COMMIT`, needs Pillow) is in the
  git history of `xnnpack/export.py` should it be worth revisiting.
- Quantizer trap: `XNNPACKQuantizer.set_module_name` honours only one name per
  quantizer (78 names -> 3 observers) and `None` configs are unsupported; to
  exclude modules set `quantizer.filter_fn` (combined with every annotator) and
  use `set_global`.
