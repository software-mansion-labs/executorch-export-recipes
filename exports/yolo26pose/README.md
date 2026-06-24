# YOLO26 Pose — ExecuTorch Export

Exports **YOLO26 Pose** keypoint detection models to `.pte` (ExecuTorch XNNPACK) format.
Produces one `.pte` file per image size, each with a single `forward()` method.

YOLO26 uses end-to-end (NMS-free) inference — the top-K detections are selected
by the model head itself, so no post-processing is required at runtime.

## Output Contract

Each `.pte` exposes a `forward(x)` taking `(1, 3, img_size, img_size)` fp32 and returning:

| Output | Shape | Description |
| --- | --- | --- |
| `boxes` | `(K, 4)` | Bounding boxes in `[xmin, ymin, xmax, ymax]` absolute pixel coords |
| `scores` | `(K,)` | Confidence scores in `[0, 1]` |
| `keypoints` | `(K, 17, 3)` | COCO-17 keypoints `(x, y, visibility)` in absolute pixel coords |

Where `K = 300` (fixed by the YOLO26 head — not configurable at export time).

## Files

| File | Description |
| --- | --- |
| `export_yolo26_pose.py` | Export script — wraps the model, traces with `torch.export`, lowers to XNNPACK, verifies parity. |
| `pyproject.toml` | Project dependencies. |
| `uv.lock` | Locked dependency graph. |
| `.python-version` | Pins Python `3.12`. |
| `.gitignore` | Ignores `.pte` outputs, `.venv/`, and caches. |

## Setup & Export

### Export all sizes (384, 512, 640)

```bash
uv run python export_yolo26_pose.py
```

This downloads `yolo26n-pose.pt` automatically on first run and produces:

```
yolo26n_pose_384_xnnpack_fp32.pte
yolo26n_pose_512_xnnpack_fp32.pte
yolo26n_pose_640_xnnpack_fp32.pte
```

### Options

```bash
uv run python export_yolo26_pose.py \
    --model-path yolo26n-pose.pt \   # model variant (n/s/m/l/x)
    --output-dir . \                 # where to write .pte files
    --sizes 384 512 640              # image sizes to export
```

## Notes

- YOLO26 uses `end2end=True` and a `Pose26` detection head. Its raw output is
  `[1, 300, 57]` = `[xyxy(4), score(1), class_id(1), 17×3 keypoints]`.
  The wrapper extracts these fields and reshapes keypoints to `[300, 17, 3]`.
- Parity between the eager PyTorch model and the ExecuTorch `.pte` is verified
  automatically after each export (max difference threshold: `5e-3` for boxes
  and keypoints, `1e-4` for scores).
