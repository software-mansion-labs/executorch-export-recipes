"""Export RF-DETR Nano (detection + instance segmentation) to ExecuTorch XNNPACK format.

Outputs
-------
Detection:
    OUTPUT_DIR/xnnpack/rfdetr_nano_det_<resolution>_xnnpack_fp32.pte
    Tensors: boxes_xyxy [Q,4], scores [Q], labels [Q]

Segmentation:
    OUTPUT_DIR/xnnpack/rfdetr_nano_seg_<resolution>_xnnpack_fp32.pte
    Tensors: boxes_xyxy [Q,4], scores [Q], labels [Q], mask_logits [Q,MH,MW]

Boxes are in xyxy pixel coords of the model input (0..resolution).
Mask logits are pre-sigmoid — threshold in the pipeline.

Usage
-----
    uv run rfdetr_export.py                          # det + seg at default resolution
    uv run rfdetr_export.py --mode det               # detection only
    uv run rfdetr_export.py --mode seg               # segmentation only
    uv run rfdetr_export.py --resolution 560 640     # multiple resolutions
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

from export_patches import apply_patches

apply_patches()

DEFAULT_RESOLUTIONS = [312]


# ── Wrappers ──────────────────────────────────────────────────────────────────


class RFDETRDetectionWrapper(nn.Module):
    """Wraps RFDETRNano to output (boxes_xyxy [Q,4], scores [Q], labels [Q]).

    Boxes are xyxy in model pixel coordinates (0..resolution).
    Scores are sigmoid confidence in [0, 1].
    Labels are float32 COCO class IDs.
    """

    def __init__(self, model: nn.Module, resolution: int):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def forward(self, x: torch.Tensor):
        pred_boxes, pred_logits = self.model(x)
        scores, labels = pred_logits[0].sigmoid().max(dim=-1)
        boxes = pred_boxes[0]
        cxy = boxes[..., :2]
        wh = boxes[..., 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        bboxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution
        return bboxes, scores, labels.float()


class RFDETRSegmentationWrapper(nn.Module):
    """Wraps RFDETRSegNano to output (boxes_xyxy [Q,4], scores [Q], labels [Q], mask_logits [Q,MH,MW]).

    Boxes are xyxy in model pixel coordinates (0..resolution).
    Scores are sigmoid confidence in [0, 1].
    Labels are float32 COCO class IDs.
    Mask logits are pre-sigmoid (threshold them on-device).

    Matches the schema expected by instanceSegmentation.ts:
        output[0]: float32 [N, 4]       boxes
        output[1]: float32 [N]          scores
        output[2]: float32 [N]          class labels
        output[3]: float32 [N, MH, MW]  mask logits
    """

    def __init__(self, model: nn.Module, resolution: int):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def forward(self, x: torch.Tensor):
        # pred_boxes: [1, Q, 4], pred_logits: [1, Q, C], pred_masks: [1, Q, MH, MW]
        pred_boxes, pred_logits, pred_masks = self.model(x)

        # cxcywh → xyxy in pixel coords
        cxy = pred_boxes[..., :2]
        wh = pred_boxes[..., 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        bboxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution

        # scores + class ids
        probs = pred_logits.sigmoid()
        max_scores, class_ids = probs.max(dim=-1)

        # drop the batch dim from all outputs
        return bboxes[0], max_scores[0], class_ids[0].float(), pred_masks[0]


# ── Lowering ──────────────────────────────────────────────────────────────────


def _lower_and_save(exported, out_path: Path):
    program = to_edge_transform_and_lower(
        exported,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
    ).to_executorch()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(program.buffer)
    print(f"  Saved {out_path} ({len(program.buffer) / 1e6:.1f} MB)")


# ── Export functions ──────────────────────────────────────────────────────────


def export_detection(resolution: int, output_dir: Path):
    from rfdetr import RFDETRNano  # type: ignore

    model = RFDETRNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    model_resolution = model.model.resolution

    wrapper = RFDETRDetectionWrapper(actual, model_resolution)
    wrapper.eval()

    example = (torch.randn(1, 3, resolution, resolution),)
    print(f"  Tracing RFDETRNano @ {resolution}x{resolution}...")
    with torch.inference_mode():
        exported = torch.export.export(wrapper, args=example, strict=False)

    out_path = output_dir / "xnnpack" / f"rfdetr_nano_det_{resolution}_xnnpack_fp32.pte"
    _lower_and_save(exported, out_path)


def export_segmentation(resolution: int, output_dir: Path):
    from rfdetr import RFDETRSegNano  # type: ignore

    model = RFDETRSegNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    model_resolution = model.model.resolution

    wrapper = RFDETRSegmentationWrapper(actual, model_resolution)
    wrapper.eval()

    example = (torch.randn(1, 3, resolution, resolution),)
    print(f"  Tracing RFDETRSegNano @ {resolution}x{resolution}...")
    with torch.inference_mode():
        exported = torch.export.export(wrapper, args=example, strict=False)

    out_path = output_dir / "xnnpack" / f"rfdetr_nano_seg_{resolution}_xnnpack_fp32.pte"
    _lower_and_save(exported, out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Export RF-DETR Nano to ExecuTorch .pte")
    parser.add_argument(
        "--mode",
        choices=["det", "seg", "all"],
        default="all",
        help="Which model to export: det (detection), seg (segmentation), all (default)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        help=f"Input resolution(s) to export (default: {DEFAULT_RESOLUTIONS})",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "OUTPUT_DIR"
    successful, failed = 0, 0

    modes = ["det", "seg"] if args.mode == "all" else [args.mode]

    for resolution in args.resolution:
        for mode in modes:
            label = f"rfdetr-nano-{'detection' if mode == 'det' else 'segmentation'} @ {resolution}x{resolution}"
            print(f"\n{'='*60}\nExporting {label}\n{'='*60}")
            try:
                if mode == "det":
                    export_detection(resolution, output_dir)
                else:
                    export_segmentation(resolution, output_dir)
                successful += 1
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print(f"\nDone: {successful} ok, {failed} failed")


if __name__ == "__main__":
    main()
