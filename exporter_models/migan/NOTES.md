# migan notes

- **Layout:** `{backend}/{size}/config.json`, both locally (`xnnpack/256/`,
  `xnnpack/512/`) and in the HF repo (`xnnpack/256/migan_256_xnnpack_fp16.pte`),
  backend on top as in the spec's model-card example and kokoro; other
  multi-size repos differ, this one follows the review. `exports/migan-hf/` is
  a ready-to-upload mirror of the HF repo (root `config.json`, `README.md`,
  `LICENSE`, `<backend>/<size>/`) built from the last export.

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
  checkpoint reconstructs them (`exports/migan/cmp_faces.png`, local), which
  is the reason to ship it. XNNPACK smoke-tested on the Android emulator
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
  than 4x the cost of 256, which suggests a fixed cost worth a profiler run
  (the 12 to 14 portable `upsample_nearest2d` ops are the first suspect).
- **Delegation:** everything lands on XNNPACK except `upsample_nearest2d` (12
  ops at 256, 14 at 512), which the partitioner does not take with or without
  `extended_ops=True`. The fp16 build additionally carries 65 / 73 portable
  `_to_dim_order_copy` casts: the three fp32-I/O casts of the wrapper plus one
  zero-dim fp32 to fp16 cast per scalar constant (gains, clamp bounds), i.e.
  negligible work. They are cheap memory ops; the upstream `Upsample2d` is
  nearest-x2 followed by a `[[1,0],[0,0]]` checkerboard multiply, i.e. zero
  insertion — a depthwise `conv_transpose2d(stride=2)` would express it as one
  delegated op if profiling shows the portable fallback matters. Not yet
  profiled with the executorch-profiler (needs the event-tracer ET build).
- **Testing trap:** when checking a `.pte` from a PIL/numpy image, `permute`
  leaves the tensor non-contiguous and the ET runtime reads the raw buffer as
  contiguous NCHW — call `.contiguous()` first or the output is garbage (and
  the "parity failure" looks identical for every build you try).
- Env pins stable `torch==2.14.0` from PyPI rather than a dated nightly (the
  nightly index prunes old builds; facemesh does the same).

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
  env: these files came from `~/Documents/softu_executorch/executorch/.venv`
  (editable executorch 1.4.1+f8b41ab = the fork's `@ms/separate-backends-1.4.1`
  lineage plus the Vulkan mean/safe_idx fix, torch 2.13.0, python 3.13) via
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
  XNNPACK; forward was several times faster than XNNPACK fp32 on the same
  phone (and still clearly ahead of the fp16 now shipped) (first call and load are longer, shader compile). Vulkan is the
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
here). Inputs are real upstream example pairs (`exports/migan/profile/inputs_*.pt`,
local), 10 forwards each. Full tables in `exports/migan/profile/` (local).
Shares below are of `Method::execute` time on this Mac; absolute numbers are
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
- **What would remove it:** one XNNPACK partition, which needs the nearest
  x2 upsample inside the delegate. XNNPACK has no nearest resize node;
  `Upsample2d` is nearest-x2 followed by a zero-insertion checkerboard, which
  is a `pixel_shuffle` (depth-to-space) of `[x, 0, 0, 0]` along channels, or a
  depthwise `conv_transpose2d(stride=2)`. The conv_transpose rewrite was
  tried during the int8 bisection: one partition, but slower on the S23 than
  the split graph, so it is not the answer; pixel_shuffle / depth_to_space
  support in the XNNPACK partitioner is the candidate ExecuTorch change.
- **GEMM, not 3x3 conv**: MI-GAN's `SeparableConv2d` is depthwise 3x3 +
  pointwise 1x1, so the heavy op is the 1x1 GEMM; depthwise convs are 10%.

**CoreML fp16, compute plan** (`coreml_compute_plan.py`, compute units `ALL`
as compiled; what the CoreML runtime on this Mac would choose, iOS may differ):

| | 256 | 512 | faces_256 |
| --- | --- | --- | --- |
| ops on ANE | 274 / 324 (85%) | 283 / 370 (77%) | 274 / 324 (85%) |
| ops on GPU | 50 (15%) | 87 (23%) | 50 (15%) |
| ops on CPU | 0 | 0 | 0 |

- The split follows resolution: at 256 the first 272 ops run on the ANE and
  the last 50, the full-resolution decoder block plus the RGB output, on the
  GPU; at 512 the first full-resolution encoder block goes to the GPU as
  well and the middle of the network stays on the ANE. Nothing on the CPU.
  Matches the on-device sweep, where `ALL` and `CPU_AND_NE` were
  indistinguishable: the ANE carries the bulk, the GPU takes the
  full-resolution ends.

**Vulkan**: the profiler drives the host runtime and has no Vulkan
backend, so only the static check applies (one `VulkanBackend` call, zero
portable ops, for all three sizes).

## XNNPACK precision study: fp32 vs fp16 vs int8 (2026-10-14, for MR !31)

Review asked whether XNNPACK should ship fp32 at all. Exported fp16 and static
int8 (PT2E, per-channel weights, `XNNPACKQuantizer`) alongside fp32 and measured
them on upstream's example sets. int8 is calibrated on the free-form sets
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
  blue speckles in smooth regions (sky, haze) and along high-contrast edges,
  see `exports/migan/cmp_volcano_int8b.png`; PSNR to fp32 inside the hole
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
  `xnnpack-weights-cache-quant-names` in the SWM fork (worktree
  `~/Documents/softu_executorch/executorch-wcfix`), with a gtest. Verified:
  cached host runner bit-identical to uncached for int8 256/512, the
  single-partition variant and the minimal U-Net repro, and the patched
  `executor_runner` on the Android arm64 emulator matches the host. Not yet
  in an RNE native-libs release, so the shipped Android lib still has it.
  Upstream has the same cache design, PR candidate.
- **Decision (2026-10-15): XNNPACK ships fp16 only.** fp32 is dropped (same
  output, slower) and int8 is dropped (quality loss on host; the Android
  divergence is a runtime bug, fixed in the fork, see above). The int8 recipe (PT2E, `XNNPACKQuantizer`, per-channel weights,
  calibration on a sparse clone of upstream `examples/` at `SOURCE_COMMIT`,
  needs Pillow) lives in this branch's history if the runtime bug gets fixed.
- Eval tooling: `exports/migan/eval_variants.py`, device runner via the local smoke-test screen + `autorun.json` (local).
- Quantizer trap: `XNNPACKQuantizer.set_module_name` honours only one name per
  quantizer (78 names -> 3 observers) and `None` configs are unsupported; to
  exclude modules set `quantizer.filter_fn` (combined with every annotator) and
  use `set_global`.
