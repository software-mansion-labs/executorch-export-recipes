# YOLOv8 Pose — ExecuTorch Export

Exports the **YOLOv8 Pose** (keypoint detection) model to `.pte` (ExecuTorch) format with post-processing (anchor decoding) bundled inside the model.

This directory contains the export scripts and configuration using the `uv` pattern.

## Output Contract

The exported `.pte` program exposes a single `forward` method with the following contract:

- **Input**: A batch of 1 image of shape `(1, 3, img_size, img_size)`. (Default input resolution is 640×640, but configurable).
- **Outputs**:
  
  | Output Index | Name | Shape | Description |
  | --- | --- | --- | --- |
  | 0 | `boxes` | `(K, 4)` | Bounding boxes in `[xmin, ymin, xmax, ymax]` absolute pixel coordinates (0 to `img_size`). |
  | 1 | `scores` | `(K,)` | Confidence scores in `[0, 1]` for each detection. |
  | 2 | `keypoints` | `(K, 17, 3)` | 17 COCO keypoints in `[x, y, confidence]` format. `x, y` are in absolute pixel coordinates (0 to `img_size`). |

Where `K` is the number of predictions kept (configurable via `--topk`, defaults to 300).

## Files

| File | Description |
| --- | --- |
| `export_yolo_pose.py` | Exporter script — defines the wrapper, traces with `torch.export`, optimizes with XNNPACK, and saves to `.pte`. |
| `yolov8n-pose.pt` | Eager PyTorch model weights from Ultralytics YOLOv8. |
| `pyproject.toml` | Project configuration and dependencies. |
| `uv.lock` | Compiled/locked project dependencies. |
| `.python-version` | Pins the Python interpreter version (`3.12`). |
| `.gitignore` | Ignores local export artifacts (`.pte`) and virtual environments (`.venv/`). |

## Setup & Export

### 1. Requirements

Ensure you have `uv` installed. The dependencies (`torch`, `executorch`, `ultralytics`) will be resolved and fetched automatically.

### 2. Export Model

Run the exporter using `uv run`:

```bash
uv run python export_yolo_pose.py
```

### Options

The exporter script supports several configuration options:

```bash
uv run python export_yolo_pose.py \
    --model-path yolov8n-pose.pt \
    --output-path yolov8n_pose_decoded.pte \
    --img-size 640 \
    --topk 300 \
    --no-xnnpack
```

- `--model-path`: Path to the input PyTorch model weights (defaults to `yolov8n-pose.pt`).
- `--output-path`: Target output file path (defaults to `yolov8n_pose_decoded.pte`).
- `--img-size`: Resolution width/height (defaults to `640`).
- `--topk`: If > 0, limits output to the top K highest scoring detections (defaults to `300`).
- `--no-xnnpack`: Disables XNNPACK delegate optimization.
