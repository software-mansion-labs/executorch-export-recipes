# migan notes

- **Layout:** `{size}/{backend}/config.json` (`256/xnnpack/`, `512/xnnpack/`),
  matching the SWM HF repo layout; `exports/migan-hf/` is a ready-to-upload
  mirror of the HF repo (root `config.json`, `README.md`, `LICENSE`, per-size
  dirs) built from the last export.

Tracks react-native-executorch issue #1373 (see `docs/migan-inpainting-plan.md`
for the full plan and the app-side hand-off).

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
- **Sizes:** `256` and `512`, both the Places2 (scenery) checkpoints. The FFHQ
  face checkpoint (`migan_256_ffhq.pt`, hash in `_migan.py`) shares the 256
  architecture and could ship as a second 256 variant later.
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
- **Verified 2026-10-13** (ET 1.4.1 runtime on host, torch 2.14.0): both `.pte`s
  report the config's contract via `method_meta`; `.pte` vs eager on upstream
  example images matches to max abs 3e-6 (256) / 3e-6 (512); visual check on
  `places2_512_object/1.png` removes the sheep cleanly. Host CPU forward:
  ~64 ms (256), ~126 ms (512).
- **On device (2026-10-13, Samsung Galaxy S23 / SM-S911B, Android 16, debug build of
  the react-native-executorch computer-vision example app, low-level `loadModel` +
  `execute`):** both `.pte`s load, report `XnnpackBackend` for `forward`, and fill a
  synthetic hole. Forward avg of 10: **460 ms (256), 848 ms (512)**; first call
  531 / 1201 ms; load 52 / 176 ms. The 512 vs 256 ratio is only 1.8x for 4x the
  pixels, so a fixed cost dominates; worth a profiler run (the 12 to 14 portable
  `upsample_nearest2d` ops are the first suspect) before deciding on int8.
- **Delegation:** everything lands on XNNPACK except `upsample_nearest2d` (12
  ops at 256, 14 at 512), which the partitioner does not take with or without
  `extended_ops=True`. They are cheap memory ops; the upstream `Upsample2d` is
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
  zero portable ops, I/O matches the config. **Not run on a device yet** -- the
  pip wheel cannot init the CoreML delegate on macOS (see
  `docs/executorch-1.4.1-bump.md`). `compute_unit` is left at the default
  `ALL` -- **swept on device, keep it** (below).
- **CoreML on device (2026-10-13, iPhone 16 / iOS 26.6, same example app,
  low-level `loadModel` + `execute`, 10-forward average, fp32 PyTorch
  reference mean-of-output 0.56661 (512) / 0.54211 (256) on the screen's
  synthetic input):**

  | compute unit | 512 ms | 512 mean | 256 ms | 256 mean |
  | --- | --- | --- | --- | --- |
  | `ALL` (default, shipped) | **35.1** | 0.56674 | **13.7** | 0.54211 |
  | `CPU_AND_NE` | 34.9 | 0.56674 | 13.8 | 0.54211 |
  | `CPU_AND_GPU` | 51.9 | 0.56666 | 27.6 | 0.54202 |
  | `CPU_ONLY` | 145.1 | 0.56664 | 60.7 | 0.54213 |
  | XNNPACK fp32 (same phone) | 712.6 | 0.56661 | 384.4 | 0.54211 |

  Every CoreML variant is numerically fine (fp16-level deltas of ~1e-4 from the
  fp32 reference; XNNPACK matches it exactly). `ALL` lands on the Neural
  Engine (identical to `CPU_AND_NE`) and is the fastest, 20x XNNPACK -- the
  opposite of style_transfer's ANE pathology, which is why the sweep is per
  model. First forward 40 ms, load 0.1-1.9 s (CoreML compile, once per
  install). No compute_unit pin needed.
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
- **Vulkan on device (2026-10-13, Galaxy S23 / Adreno, same app and harness as
  the XNNPACK numbers):** both sizes load (`VulkanBackend`) and fill the
  synthetic hole with the same mean as XNNPACK (0.494 / 0.495). Forward avg of
  10: **139 ms (256), 229 ms (512)** vs 460 / 848 ms on XNNPACK, i.e. 3.3x /
  3.7x faster; first call 217 / 306 ms; load 79 / 503 ms (shader compile).
  Vulkan is the Android default candidate for this model.
- **Download cache trap when re-testing on device:** the RNE fetcher keys its
  cache on the URL path (a `?v=2` query does not bust it), so a re-exported
  file at the same URL is never re-downloaded. Delete
  `/storage/emulated/0/Android/data/<app>/files/react-native-executorch/*` via
  adb, or rename the file.

## XNNPACK precision study: fp32 vs fp16 vs int8 (2026-10-14, for MR !31)

Review asked whether XNNPACK should ship fp32 at all. Exported fp16 and static
int8 (PT2E, per-channel weights, `XNNPACKQuantizer`) alongside fp32 and measured
them on upstream's example sets. int8 is calibrated on the free-form sets
(`places2_512_freeform` for 512; that plus `places2_256_freeform` for 256) and
evaluated on sets it never saw: `places2_512_object` and `ffhq_256_freeform`
(faces, out of domain, random masks). Metrics are computed **inside the hole
only**, against the fp32 export's output and against the original pixels;
"PSNR vs GT" is only meaningful on random-mask sets (on the object set the
original still contains the removed object). Host = this Mac, median forward.

| 512 | MB | host ms | PSNR vs fp32 | SSIM vs fp32 | mean abs diff | PSNR vs GT (ffhq) |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 28.0 | 118 | - | 1.0000 | 0 | 17.64 |
| fp16 | 25.9 | 93 | 55 dB | 0.9999 | 0.001 | 17.65 |
| int8 | 7.3 | 76 | 25.4 dB (min 21) | 0.937 | 0.034 | 17.10 |
| int8, RGB path kept fp32 | 8.8 | 80 | 24.1 dB | 0.927 | 0.040 | 16.99 |

| 256 | MB | host ms | PSNR vs fp32 | SSIM vs fp32 | mean abs diff | PSNR vs GT (ffhq) |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 24.7 | 64 | - | 1.0000 | 0 | 17.80 |
| fp16 | 24.2 | 54 | 56 dB | 0.9999 | 0.001 | 17.80 |
| int8 | 6.5 | 45 | 26.6 dB (min 21) | 0.949 | 0.030 | 17.55 |

- **On device (S23, same screen, avg of 10 forwards, 2026-10-14):**

| S23 XNNPACK | fp32 | fp16 | int8 |
| --- | --- | --- | --- |
| 256 | 440 ms | 213 ms | 170 ms (wrong output) |
| 512 | 809 ms | 410 ms | 336 ms (wrong output) |

- **fp16 is numerically fp32** (55 dB, mean abs diff ~1/1000) and only 8%
  smaller (the delegate keeps most weight bytes in fp32, as facemesh records),
  but on the phone it is **~2x faster**: ARM fp16 NEON arithmetic, unlike the
  Mac where the gain was 25%. fp16 output on device matches fp32 to 4 decimals
  (mean-of-output 0.56665 vs 0.56661 at 512).
- **int8 on host is 3.8x smaller and ~1.6x faster, with visible damage**:
  blue speckles in smooth regions (sky, haze) and along high-contrast edges,
  see `exports/migan/cmp_volcano_int8b.png`; PSNR to fp32 inside the hole
  ~25 dB, i.e. ~8/255 average error, and 0.3-0.5 dB lost against ground truth.
  Keeping the RGB skip path (`fromrgb`, `torgb`, image upsample) in fp32 does
  NOT help, so the error is in the feature path -- plausibly the activation
  range (the `lrelu_agc` clamp is +/-256, so per-tensor 8-bit steps are coarse).
- **int8 on the RNE Android runtime computes the wrong thing** (libs 0.10.4,
  ET 1.4.1 headers): the same `.pte` gives mean-in-hole 0.49 on the host
  runtime but 0.80 (256) / 1.00 saturated (512) on the S23, and the Android
  emulator reproduces the S23 numbers exactly, so it is the Android build, not
  the SoC. Bisection (all on the synthetic-gradient screen input): per-tensor
  weights, or leaving mul/add/cat unquantized, change nothing; single MI-GAN
  building blocks (conv3x3, dw 4x4 + pad, lrelu/gain/clamp chains, 64 ch) are
  bit-exact host vs emulator; a minimal U-Net skip loop (encoder, portable
  `upsample_nearest2d`, multiply by the constant checkerboard, decoder) diverges
  (host 3.59 vs 2.25 mean), is bit-exact without the checkerboard multiply or
  with the whole upsample path as one `conv_transpose2d`, and still diverges
  when that multiply is left in fp32. Folding MI-GAN's `Upsample2d` into a
  depthwise `conv_transpose2d(stride=2, padding=1)` (eager parity 3e-7, one
  XNNPACK partition) shrinks the error (0.60) but does not remove it, and is
  slower for fp32 on the S23 (622 vs 440 ms), so it is not a fix. Repro files:
  `exports/migan/migan_256_micro_skip*.pte` (local). Root cause not pinned;
  next step would be a layer-by-layer bisection of the generator.
- **Decision (2026-10-15): XNNPACK ships fp16 only.** fp32 is dropped (same
  output, half the speed) and int8 is dropped (quality loss on host, wrong on
  Android). The int8 recipe (PT2E, `XNNPACKQuantizer`, per-channel weights,
  calibration on a sparse clone of upstream `examples/` at `SOURCE_COMMIT`,
  needs Pillow) lives in this branch's history if the runtime bug gets fixed.
- Eval tooling: `exports/migan/eval_variants.py`, device runner via the local smoke-test screen + `autorun.json` (local).
- Quantizer trap: `XNNPACKQuantizer.set_module_name` honours only one name per
  quantizer (78 names -> 3 observers) and `None` configs are unsupported; to
  exclude modules set `quantizer.filter_fn` (combined with every annotator) and
  use `set_global`.
