# RF-DETR Keypoints — ExecuTorch Export

Exports the **RF-DETR keypoint preview** model (single-person, COCO-17 pose, 576×576
input) to `.pte` (ExecuTorch) for on-device inference on Apple Silicon.

The model (`RFDETRKeypointPreview`) ships in upstream
[RF-DETR](https://github.com/roboflow/rf-detr) — `pip install rfdetr` — so no fork is
needed; this directory only adds the ExecuTorch export logic on top of it.

`export_keypoint_ptes.sh` produces, in order:

| # | File | Backend / precision | Latency (M4 Pro) | Accuracy vs PyTorch | Size |
|---|---|---|---|---|---|
| 1 | `rfdetr_keypoint_preview_coreml_fp16_cpu_and_gpu.pte` | **CoreML fp16, GPU — recommended** | **~42 ms** | sub-pixel (≤0.7 px) | 72 MB |
| 2 | `rfdetr_keypoint_preview_coreml_fp32.pte` | CoreML fp32, GPU (bit-exact fallback) | ~47 ms | 0.00 px | 141 MB |
| 3 | `rfdetr_keypoint_preview_xnnpack_fp32.pte` | XNNPACK (portable CPU) | ~330 ms | 0.00 px | 139 MB |
| 4 | `rfdetr_keypoint_preview_mlx_fp32.pte` | MLX (Apple GPU) — opt-in `MLX=1` | ~66 ms | sub-pixel | 139 MB |

The CoreML **fp16/GPU** build is the one to ship — fastest with negligible accuracy
loss. fp32 is a bit-exact fallback; XNNPACK is for non-Apple portability only.

## Files

| File | What it is |
|---|---|
| `export.py` | The actual exporter — defines the output wrappers, traces with `torch.export`, and lowers to CoreML / XNNPACK / MLX. Run it directly or via the script below. |
| `export_patches.py` | `torch.export` compatibility patches for RF-DETR's deformable attention / positional encoding (bundled copy, applied at runtime to the installed `rfdetr`). |
| `export_keypoint_ptes.sh` | Convenience runner — calls `export.py` once per backend with the recommended settings. |
| `README.md` | This file. |

Only the **model classes** come from the `rfdetr` package; the export logic lives here.

## Dependencies

You need a Python environment with the model package and the export toolchain:

| Dependency | Notes |
|---|---|
| `rfdetr` | Provides `RFDETRKeypointPreview` (keypoint preview, pretrained on COCO person keypoints). **It ships in upstream RF-DETR — just `pip install rfdetr`** ([`roboflow/rf-detr`](https://github.com/roboflow/rf-detr)); you do **not** need a fork. (This export work was developed against the `feat/key-points` branch of `rf-detr-keypoints`; that checkout's `.venv` works too and is required for the MLX build.) |
| `executorch`, `coremltools` (≥8), `torch`, `tabulate`, `numpy` | The export/lowering toolchain. |
| `flatc` | FlatBuffers compiler for XNNPACK serialization — must be on `PATH`. Lives at `<venv>/bin/flatc`; the script adds it automatically. |
| ExecuTorch built from source **(MLX only)** | The MLX backend is not in the pip wheel. Only needed if you pass `MLX=1`. See `dependencies/executorch.md` and the model repo's `memory/mlx-export-build.md`. |

The simplest setup for the CoreML/XNNPACK path is a fresh env:

```bash
pip install rfdetr executorch coremltools tabulate    # rfdetr pulls in torch/torchvision
PYTHON=$(which python) ./export_keypoint_ptes.sh
```

The **rf-detr-keypoints checkout's `.venv`** is the other turnkey option (it already has
everything, including the source-built MLX ExecuTorch). The runner's `REPO=` points at
that checkout purely to find its `.venv`; use `PYTHON=` to point at any other env.

> Note: `export_patches.py` monkey-patches `rfdetr` model internals for `torch.export`,
> so use a reasonably current `rfdetr` — if a future release changes those internals the
> patches may need a refresh.

## Usage

```bash
# Default: CoreML fp16/GPU + CoreML fp32 + XNNPACK, into ./exports
REPO=/path/to/rf-detr-keypoints ./export_keypoint_ptes.sh

# Also export the MLX backend (requires a source-built executorch with the MLX delegate)
REPO=/path/to/rf-detr-keypoints MLX=1 ./export_keypoint_ptes.sh

# Custom output dir / Python interpreter
REPO=... OUT=/tmp/pte PYTHON=/path/to/venv/bin/python ./export_keypoint_ptes.sh
```

`REPO` defaults to `/Users/jakubchmura/Desktop/SWM_AI/rf-detr-keypoints`; override `REPO`
(or `PYTHON`) for your machine. Outputs land in `./exports/` next to the script (gitignored).

To export a single backend / tweak settings, call `export.py` directly with an env that
has the deps (run from this directory so `export_patches.py` is importable):

```bash
/path/to/venv/bin/python export.py \
    --variant RFDETRKeypointPreview --backend coreml \
    --compute-unit cpu_and_gpu --compute-precision fp16 \
    --output-dir exports --output-name rfdetr_keypoint_preview_coreml_fp16_cpu_and_gpu
# other useful flags: --resolution N, --coreml-quant {int8,int4}, --min-deployment-target,
# --compute-unit {all,cpu_and_ne,cpu_and_gpu,cpu_only}, --xnnpack-quant {dynamic,static}
```

## Output contract

Each `.pte` exposes a `forward(x)` taking a normalized `(1, 3, 576, 576)` fp32 tensor
(ImageNet mean/std) and returning three tensors over `Q = 100` query slots:

| Output | Shape | Meaning |
|---|---|---|
| `boxes` | `(Q, 4)` | person box `xyxy`, in **model-input pixels** (0–576) |
| `scores` | `(Q,)` | person confidence in `[0, 1]` (sigmoid) |
| `keypoints` | `(Q, 17, 3)` | per-detection `(x, y, visibility)`; `x, y` in model-input pixels, `visibility` in `[0, 1]` |

Coordinates are in the 576×576 model space — scale by `orig_w / 576` and
`orig_h / 576` to map back to your source image. Filter detections by `scores`, then
draw keypoints whose `visibility ≥ ~0.5`. A ready-made `predict()`-style wrapper lives
in the model repo at `src/rfdetr/export/executorch_runtime.py`
(`ExecuTorchKeypointPredictor`).

## How it works

All of the below is in `export.py` (steps 2–5) and `export_patches.py` (step 3):

1. **Load** `RFDETRKeypointPreview`, move to CPU, call `.export()` (export-mode forward).
2. **Wrap** with `RFDETRKeypointWrapper` to produce the `(boxes, scores, keypoints)`
   contract above in pixel space.
3. **Patch** the deformable attention / positional encoding for `torch.export`
   compatibility (rank-5 reshape for CoreML; a grid-free sampling core for MLX).
4. **`torch.export.export`** with a random `(1, 3, 576, 576)` example.
5. **Lower** to the backend: CoreML (`CoreMLPartitioner`), XNNPACK (`XnnpackPartitioner`
   + `AddmmToLinearTransform`), or MLX (`MLXPartitioner`), then serialize the `.pte`.

## Findings

Benchmarked on an M4 Pro (June 2026). The harnesses live in the model repo:
`benchmark_pte.py` (latency + drift) and `validate_predictor.py` (keypoint accuracy).

- **The model is GPU-bound, not ANE-bound.** Forcing the ANE (`--compute-unit
  cpu_and_ne`) is ~197 ms — its deformable attention isn't ANE-friendly. CoreML `all`
  already schedules it on the GPU (~47 ms).
- **fp16 on the GPU is the win** (~42 ms, ≤0.7 px drift). Note `fp16 + all` is a trap
  (~71 ms — it tries the slow ANE); you must pin to the GPU (`--compute-unit cpu_and_gpu`).
- **Quantization does not help latency.** Weight-only int8/int4 only add runtime
  dequant on the GPU, so they're *slower* than fp32 (int4 ≈ 49 ms) and wreck keypoint
  accuracy. int4 is only useful to shrink the file (19 MB), not for speed.
- **`--min-deployment-target iOS17/iOS18` BREAKS the keypoint head** (detections vanish,
  even at 576). The default target lowers correctly — the script keeps it.
- **Lowering resolution speeds inference but breaks keypoints** without retraining: the
  positional-encoding interpolation shifts keypoints ~100 px (person *detection* still
  survives, so a low-res model is fine for detection-only/tracking passes).
