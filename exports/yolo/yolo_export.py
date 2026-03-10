"""Export YOLO26 variants to ExecuTorch XNNPACK format (multi-method)."""

from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export
from ultralytics import YOLO
from wrappers import YOLOInstanceSegmentation, YOLOObjectDetection

IMAGE_SIZES = [
    (384, 384, "forward_384"),
    (512, 512, "forward_512"),
    (640, 640, "forward_640"),
]

SIZES = ["n", "s", "m", "l", "x"]
TYPES = [
    ("", "detection"),
    ("-seg", "segmentation"),
]


def export_model_for_size(model, height, width):
    inputs = (torch.randn(1, 3, height, width),)
    model(*inputs)
    return export(model, inputs)


def export_yolo(model_name, output_path, is_segmentation):
    yolo_model = YOLO(model_name)

    if is_segmentation:
        model = YOLOInstanceSegmentation(
            yolo_model.model, method_size=IMAGE_SIZES[0][0]
        )
    else:
        model = YOLOObjectDetection(yolo_model.model, method_size=IMAGE_SIZES[0][0])

    exported_programs = {}
    for height, width, method_name in IMAGE_SIZES:
        print(f"  {method_name} ({height}x{width})")
        exported_programs[method_name] = export_model_for_size(model, height, width)

    print("Lowering to XNNPACK...")
    et_prog = to_edge_transform_and_lower(
        exported_programs, partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_prog.buffer)

    print(f"Saved {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return True


def main():
    outputs_dir = Path(__file__).parent / "OUTPUT_DIR"
    successful, failed = 0, 0

    for suffix, type_name in TYPES:
        for size in SIZES:
            name = f"yolo26{size}{suffix}"
            output_path = outputs_dir / "xnnpack" / f"{name}.pte"

            print(f"\n{'='*60}\nExporting {name}\n{'='*60}")
            try:
                export_yolo(
                    f"{name}.pt", output_path, is_segmentation=(suffix == "-seg")
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
