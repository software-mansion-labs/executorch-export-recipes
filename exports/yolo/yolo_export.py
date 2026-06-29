"""Export YOLO26 variants to ExecuTorch XNNPACK format (one file per image size)."""

from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export
from ultralytics import YOLO
from wrappers import YOLOInstanceSegmentation, YOLOObjectDetection

IMAGE_SIZES = [
    (384, 384),
    (512, 512),
    (640, 640),
]

SIZES = ["n", "s", "m", "l", "x"]
TYPES = [
    ("-seg", "segmentation"),
]


def export_yolo_for_size(yolo_model, height, width, output_path, is_segmentation):
    if is_segmentation:
        model = YOLOInstanceSegmentation(
            yolo_model.model, method_size=height
        )
    else:
        model = YOLOObjectDetection(yolo_model.model, method_size=height)

    inputs = (torch.randn(1, 3, height, width),)
    model(*inputs)
    exported = export(model, inputs)

    print("  Lowering to XNNPACK...")
    et_prog = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_prog.buffer)

    print(f"  Saved {output_path} ({len(et_prog.buffer) / 1e6:.1f} MB)")
    return True


def main():
    outputs_dir = Path(__file__).parent / "OUTPUT_DIR"
    successful, failed = 0, 0

    for suffix, type_name in TYPES:
        for size in SIZES:
            name = f"yolo26{size}{suffix}"

            print(f"\n{'='*60}\nExporting {name}\n{'='*60}")
            try:
                yolo_model = YOLO(f"{name}.pt")
                for height, width in IMAGE_SIZES:
                    output_path = outputs_dir / "xnnpack" / f"{name}_{height}_xnnpack_fp32.pte"
                    print(f"--- Exporting {name} ({height}x{width}) -> {output_path}")
                    export_yolo_for_size(
                        yolo_model, height, width, output_path, is_segmentation=(suffix == "-seg")
                    )
                successful += 1
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback

                traceback.print_exc()
                failed += 1

    print(f"\nDone: {successful} ok, {failed} failed")


if __name__ == "__main__":
    main()
