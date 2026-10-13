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
