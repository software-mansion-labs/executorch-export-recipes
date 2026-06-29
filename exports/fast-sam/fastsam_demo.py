"""FastSAM ExecuTorch demo — point and box prompts.

The model must be exported first:
    python export_fastsam.py --size x

Usage:
    python fastsam_demo.py zidane.jpg
    python fastsam_demo.py zidane.jpg --model-path output/fastsam-x/xnnpack/fastsam-x.pte

Controls:
    Left-click          point prompt  — segments the object under the cursor
    Right-drag          box prompt    — segments the best-matching object in the box
    q                   quit
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from executorch.runtime import Runtime

DEFAULT_MODEL = Path("output/fastsam-x/coreml/fastsam-x_fp16.pte")
DEFAULT_MODEL = Path("output/fastsam-x/xnnpack/fastsam-x.pte")
DEFAULT_IMAGE = Path("assets/cat.jpg")
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class FastSAMExecuTorch:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        runtime = Runtime.get()
        program = runtime.load_program(str(model_path))
        self.method = program.load_method("forward")
        if self.method is None:
            raise RuntimeError("Model has no 'forward' method.")

    def run(self, image: np.ndarray):
        """Run inference with timing. Returns detections and timing dict."""
        t_preproc_start = time.perf_counter()
        tensor, ratio_pad = _preprocess(image)
        t_preproc = time.perf_counter() - t_preproc_start
        
        t_infer_start = time.perf_counter()
        bbox, conf, classes, mask_logits = self.method.execute((tensor,))
        t_infer = time.perf_counter() - t_infer_start
        
        timing = {"preprocess_ms": t_preproc * 1000, "inference_ms": t_infer * 1000}
        return bbox, conf, mask_logits, ratio_pad, timing


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _letterbox(image: np.ndarray, new_shape=INPUT_SIZE, color=(114, 114, 114)):
    h, w = image.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * ratio)), int(round(h * ratio)))
    pad_w = new_shape[1] - new_unpad[0]
    pad_h = new_shape[0] - new_unpad[1]
    pad_left = int(round(pad_w / 2 - 0.1))
    pad_top  = int(round(pad_h / 2 - 0.1))
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    letterboxed = cv2.copyMakeBorder(
        resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=color,
    )
    return letterboxed, ratio, (pad_left, pad_top)


def _preprocess(image: np.ndarray):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lb, ratio, pad = _letterbox(rgb)
    tensor = torch.from_numpy(
        np.ascontiguousarray(lb.transpose(2, 0, 1))
    ).float().unsqueeze(0) / 255.0
    return tensor, (ratio, pad)


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def _scale_boxes(boxes: torch.Tensor, ratio_pad, img0_shape) -> torch.Tensor:
    """Map boxes from model-input pixel space → original image pixel space."""
    ratio, (pad_x, pad_y) = ratio_pad
    boxes = boxes.clone()
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= ratio
    h0, w0 = img0_shape[:2]
    boxes[:, 0].clamp_(0, w0)
    boxes[:, 1].clamp_(0, h0)
    boxes[:, 2].clamp_(0, w0)
    boxes[:, 3].clamp_(0, h0)
    return boxes


def _scale_masks(masks: torch.Tensor, img0_shape) -> torch.Tensor:
    """Upsample proto-resolution masks to original image size, removing letterbox padding."""
    mh, mw = masks.shape[1:]
    h0, w0 = img0_shape[:2]
    gain = min(mh / h0, mw / w0)
    pad_w = (mw - round(w0 * gain)) / 2
    pad_h = (mh - round(h0 * gain)) / 2
    top  = round(pad_h - 0.1)
    left = round(pad_w - 0.1)
    bottom = mh - round(pad_h + 0.1)
    right  = mw - round(pad_w + 0.1)
    cropped = masks[:, top:bottom, left:right].unsqueeze(0).float()
    return F.interpolate(cropped, size=(h0, w0), mode="bilinear", align_corners=False)[0]


def _crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Zero out each mask outside its bounding box."""
    n, h, w = masks.shape
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cols = torch.arange(w, device=masks.device, dtype=boxes.dtype)[None, None, :]
    rows = torch.arange(h, device=masks.device, dtype=boxes.dtype)[None, :, None]
    return masks * (
        (cols >= x1[:, None, None]) & (cols < x2[:, None, None]) &
        (rows >= y1[:, None, None]) & (rows < y2[:, None, None])
    )


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ix1 = boxes[rest, 0].clamp(min=boxes[i, 0])
        iy1 = boxes[rest, 1].clamp(min=boxes[i, 1])
        ix2 = boxes[rest, 2].clamp(max=boxes[i, 2])
        iy2 = boxes[rest, 3].clamp(max=boxes[i, 3])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        area_i = (boxes[i, 2] - boxes[i, 0]).clamp(0) * (boxes[i, 3] - boxes[i, 1]).clamp(0)
        area_r = (boxes[rest, 2] - boxes[rest, 0]).clamp(0) * (boxes[rest, 3] - boxes[rest, 1]).clamp(0)
        iou = inter / (area_i + area_r - inter + 1e-7)
        order = rest[iou <= iou_threshold]
    return torch.stack(keep) if keep else boxes.new_zeros(0, dtype=torch.long)


def predict(model: FastSAMExecuTorch, image: np.ndarray) -> tuple[list[dict], dict]:
    t_total_start = time.perf_counter()
    bbox, scores_tensor, mask_logits, ratio_pad, model_timing = model.run(image)

    t_postproc_start = time.perf_counter()
    conf = scores_tensor
    keep = conf > CONF_THRESHOLD
    bbox, conf, mask_logits = bbox[keep], conf[keep], mask_logits[keep]
    if bbox.shape[0] == 0:
        timing = {
            "preprocess_ms": model_timing["preprocess_ms"],
            "inference_ms": model_timing["inference_ms"],
            "postprocess_ms": (time.perf_counter() - t_postproc_start) * 1000,
            "total_ms": (time.perf_counter() - t_total_start) * 1000,
        }
        return [], timing

    keep_idx = _nms(bbox, conf, IOU_THRESHOLD)
    bbox       = bbox[keep_idx]
    conf       = conf[keep_idx]
    mask_logits = mask_logits[keep_idx]

    bbox_orig = _scale_boxes(bbox, ratio_pad, image.shape)

    probs = torch.sigmoid(mask_logits)           # [N, Hm, Wm]
    masks = _scale_masks(probs, image.shape)     # [N, H0, W0]
    masks = _crop_mask(masks, bbox_orig)         # [N, H0, W0]
    masks = masks.gt(0.5).numpy()                # bool [N, H0, W0]

    bbox_orig = bbox_orig.numpy()
    conf      = conf.numpy()

    timing = {
        "preprocess_ms": model_timing["preprocess_ms"],
        "inference_ms": model_timing["inference_ms"],
        "postprocess_ms": (time.perf_counter() - t_postproc_start) * 1000,
        "total_ms": (time.perf_counter() - t_total_start) * 1000,
    }

    return [
        {"box": bbox_orig[i], "score": float(conf[i]), "mask": masks[i]}
        for i in range(len(bbox_orig))
    ], timing


# ---------------------------------------------------------------------------
# Prompt selection
# ---------------------------------------------------------------------------

def _mask_area(det):
    return int(det["mask"].sum())

def _box_area(box):
    return max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)


def select_by_point(detections: list[dict], x: int, y: int) -> dict | None:
    matches = [d for d in detections if d["mask"][y, x]]
    if not matches:
        return None
    return min(matches, key=lambda d: (_mask_area(d), _box_area(d["box"]), -d["score"]))


def select_by_box(detections: list[dict], prompt_box: np.ndarray) -> dict | None:
    px1, py1, px2, py2 = prompt_box
    prompt_area = max(px2 - px1, 0.0) * max(py2 - py1, 0.0)
    matches = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        ix1, iy1 = max(px1, x1), max(py1, y1)
        ix2, iy2 = min(px2, x2), min(py2, y2)
        inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
        if inter <= 0:
            continue
        union = prompt_area + _box_area(d["box"]) - inter + 1e-7
        matches.append((inter / union, d))
    if not matches:
        return None
    return max(matches, key=lambda t: (t[0], -_mask_area(t[1]), t[1]["score"]))[1]


# ---------------------------------------------------------------------------
# Interactive OpenCV app
# ---------------------------------------------------------------------------

def _render(state: dict) -> np.ndarray:
    frame = state["image"].copy()
    if state.get("box_active") and state.get("box_start") and state.get("box_end"):
        cv2.rectangle(frame, state["box_start"], state["box_end"], (0, 255, 255), 2)
    return frame


def _mouse_callback(event, x, y, _flags, state: dict):
    image      = state["image"]
    detections = state["detections"]

    if event == cv2.EVENT_LBUTTONDOWN:
        det = select_by_point(detections, x, y)
        if det is not None:
            cutout = np.zeros_like(image)
            cutout[det["mask"]] = image[det["mask"]]
            cv2.imshow("Segmented Cutout", cutout)
        else:
            print(f"No object at ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        state["box_start"] = (x, y)
        state["box_end"]   = (x, y)
        state["box_active"] = True

    elif event == cv2.EVENT_MOUSEMOVE and state.get("box_active"):
        state["box_end"] = (x, y)

    elif event == cv2.EVENT_RBUTTONUP and state.get("box_active"):
        state["box_active"] = False
        start = state.pop("box_start", None)
        state.pop("box_end", None)
        if start is None:
            return
        x0, y0 = start
        prompt_box = np.array([
            min(x0, x), min(y0, y),
            max(x0, x), max(y0, y),
        ], dtype=np.float32)
        det = select_by_box(detections, prompt_box)
        if det is not None:
            cutout = np.zeros_like(image)
            cutout[det["mask"]] = image[det["mask"]]
            cv2.imshow("Segmented Cutout", cutout)
        else:
            print("No object in box.")


def run_app(image_path: Path, model_path: Path) -> None:
    model = FastSAMExecuTorch(model_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    print("Running inference...")
    detections, timing = predict(model, image)
    print(f"Found {len(detections)} instances.")
    print(f"\nTiming breakdown:")
    print(f"  Preprocess:    {timing['preprocess_ms']:7.2f} ms")
    print(f"  Inference:     {timing['inference_ms']:7.2f} ms")
    print(f"  Postprocess:   {timing['postprocess_ms']:7.2f} ms")
    print(f"  ─────────────────────────")
    print(f"  Total:         {timing['total_ms']:7.2f} ms")

    state = {
        "image": image,
        "detections": detections,
        "box_active": False,
    }

    window = "FastSAM ExecuTorch — left-click: point prompt  right-drag: box prompt  q: quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _mouse_callback, param=state)

    while True:
        cv2.imshow(window, _render(state))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastSAM ExecuTorch demo")
    parser.add_argument("image_path", nargs="?", default=str(DEFAULT_IMAGE))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    run_app(Path(args.image_path), args.model_path)
