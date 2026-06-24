#!/usr/bin/env python3
"""Export YOLO26 Pose model to ExecuTorch XNNPACK format.

Produces one .pte file per image size (384, 512, 640), each exposing a single
forward() method. This matches the yolov8npose export pattern.

Output contract (per .pte):
    Input:  (1, 3, img_size, img_size)  fp32
    Output:
        boxes      (K, 4)     xyxy absolute pixel coords
        scores     (K,)       confidence in [0, 1]
        keypoints  (K, 17, 3) (x, y, visibility) in absolute pixel coords

Where K = topk (default 300). YOLO26 uses end-to-end inference (NMS-free),
so K detections are already selected by the model head.
"""
import argparse
import sys

import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLO26PoseWrapper(nn.Module):
    """Wraps a YOLO26 Pose model into the BlazeFace decoded output schema.

    YOLO26 pose head (end2end=True) outputs [B, K, 57]:
        [:, :, 0:4]  xyxy boxes, already decoded to absolute pixels
        [:, :, 4]    confidence score
        [:, :, 5]    class id
        [:, :, 6:57] 17 * 3 keypoint values (x, y, visibility)

    This wrapper squeezes the batch dim and reshapes keypoints.
    """

    def __init__(self, yolo_inner_model: nn.Module) -> None:
        super().__init__()
        self.inner = yolo_inner_model
        self.inner.eval()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.inner(x)
        preds = outputs[0].squeeze(0)  # [K, 57]

        boxes = preds[:, :4]                    # [K, 4]  xyxy
        scores = preds[:, 4]                    # [K]
        keypoints = preds[:, 6:].reshape(-1, 17, 3)  # [K, 17, 3]

        return boxes, scores, keypoints


def export_size(model: nn.Module, img_size: int, output_path: str) -> tuple:
    """Export and verify one resolution. Returns eager outputs for parity check."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    dummy_input = torch.randn(1, 3, img_size, img_size)

    # Eager run (for parity check later)
    with torch.no_grad():
        eager_boxes, eager_scores, eager_kps = model(dummy_input)

    print(f"  Eager outputs: boxes={list(eager_boxes.shape)}, "
          f"scores={list(eager_scores.shape)}, keypoints={list(eager_kps.shape)}")

    # torch.export
    print("  Exporting with torch.export...")
    with torch.no_grad():
        exported = torch.export.export(model, (dummy_input,), strict=True)

    # Lower to XNNPACK
    print("  Lowering to XNNPACK...")
    et_prog = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    # Save
    with open(output_path, "wb") as f:
        f.write(et_prog.buffer)

    size_mb = len(et_prog.buffer) / 1e6
    print(f"  Saved {output_path} ({size_mb:.1f} MB)")

    # Parity verification
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
        print("  Verifying parity...")
        et_model = _load_for_executorch_from_buffer(et_prog.buffer)
        et_outputs = et_model.run_method("forward", (dummy_input,))
        et_boxes, et_scores, et_kps = et_outputs[0], et_outputs[1], et_outputs[2]

        diff_box = torch.max(torch.abs(eager_boxes - et_boxes)).item()
        diff_cls = torch.max(torch.abs(eager_scores - et_scores)).item()
        diff_kp  = torch.max(torch.abs(eager_kps - et_kps)).item()

        print(f"  Max difference — boxes: {diff_box:.2e}, scores: {diff_cls:.2e}, keypoints: {diff_kp:.2e}")

        if diff_box < 5e-3 and diff_cls < 1e-4 and diff_kp < 5e-3:
            print("  Parity verification: PASSED ✓")
        else:
            print("  Parity verification: WARNING (larger than expected difference)")
    except Exception as e:
        print(f"  Skipping parity check: {e}")

    return eager_boxes, eager_scores, eager_kps


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO26 Pose to ExecuTorch XNNPACK format (one .pte per image size)."
    )
    parser.add_argument(
        "--model-path", type=str, default="yolo26n-pose.pt",
        help="Path to the Ultralytics YOLO26 pose model weights (default: yolo26n-pose.pt)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to write output .pte files into (default: current directory)."
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[384, 512, 640],
        help="Image sizes to export (default: 384 512 640)."
    )
    args = parser.parse_args()

    try:
        from executorch.exir import to_edge_transform_and_lower  # noqa: F401
    except ImportError:
        print("Error: executorch is not installed. Run: uv run python export_yolo26_pose.py")
        sys.exit(1)

    print(f"Loading YOLO26 pose model from {args.model_path}...")
    try:
        yolo = YOLO(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    model = YOLO26PoseWrapper(yolo.model)
    model.eval()

    stem = args.model_path.removesuffix(".pt").replace("-", "_")
    successful, failed = 0, 0

    for img_size in args.sizes:
        output_path = f"{args.output_dir}/{stem}_{img_size}_xnnpack_fp32.pte"
        sep = "=" * 60
        print(f"\n{sep}\nExporting {img_size}x{img_size} → {output_path}\n{sep}")
        try:
            export_size(model, img_size, output_path)
            successful += 1
        except Exception as e:
            import traceback
            print(f"FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nDone: {successful} ok, {failed} failed")


if __name__ == "__main__":
    main()
