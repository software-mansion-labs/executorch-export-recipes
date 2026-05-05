"""Export FastSAM variants to ExecuTorch XNNPACK format.

Usage:
    python export_fastsam.py
    python export_fastsam.py --size s --input-size 640
    python export_fastsam.py --size x --output-dir /path/to/output
"""

import argparse
from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export
from ultralytics import FastSAM

from fastsam_wrapper import FastSAMInstanceSegmentation

SIZES = {
    "s": ("FastSAM-s.pt", 640),   # FastSAM-s default input size
    "x": ("FastSAM-x.pt", 640),  # FastSAM-x default input size
}


def export_fastsam(size: str, input_size: int | None, output_dir: Path) -> None:
    model_file, default_input_size = SIZES[size]
    height = width = input_size or default_input_size
    output_path = output_dir / f"fastsam-{size}" / "xnnpack" / f"fastsam-{size}.pte"

    print(f"\n{'='*60}")
    print(f"Exporting fastsam-{size}  ({height}x{width})")
    print(f"{'='*60}")

    print(f"Loading {model_file}...")
    inner = FastSAM(model_file).model

    print("Wrapping model...")
    model = FastSAMInstanceSegmentation(inner, method_size=height)
    model.eval()

    print(f"Tracing forward ({height}x{width})...")
    inputs = (torch.randn(1, 3, height, width),)
    with torch.no_grad():
        model(*inputs)

    exported = export(model, inputs)

    print("Lowering to XNNPACK...")
    et_prog = to_edge_transform_and_lower(
        {"forward": exported},
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_prog.buffer)

    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved {output_path}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FastSAM to ExecuTorch")
    parser.add_argument(
        "--size",
        choices=list(SIZES),
        default=None,
        help="Which variant to export (default: all)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Override input resolution (square). Defaults per variant: s=640, x=1024",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Root directory for exported .pte files",
    )
    args = parser.parse_args()

    sizes = [args.size] if args.size else list(SIZES)
    successful, failed = 0, 0

    for size in sizes:
        try:
            export_fastsam(size, args.input_size, args.output_dir)
            successful += 1
        except Exception as exc:
            print(f"FAILED fastsam-{size}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nDone: {successful} ok, {failed} failed")


if __name__ == "__main__":
    main()
