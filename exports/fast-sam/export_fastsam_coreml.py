"""Export FastSAM variants to CoreML format via ExecutorTorch.

Usage:
    python export_fastsam_coreml.py
    python export_fastsam_coreml.py --size s --input-size 640
    python export_fastsam_coreml.py --size x --precision fp32
    python export_fastsam_coreml.py --output-dir /path/to/output
"""

import argparse
from pathlib import Path

import coremltools as ct
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export
from ultralytics import FastSAM

from fastsam_wrapper import FastSAMInstanceSegmentation

SIZES = {
    "s": ("FastSAM-s.pt", 640),   # FastSAM-s default input size
    "x": ("FastSAM-x.pt", 640),  # FastSAM-x default input size
}

PRECISIONS = {
    "fp32": ct.precision.FLOAT32,
    "fp16": ct.precision.FLOAT16,
}


def export_fastsam_coreml(
    size: str, input_size: int | None, precision: str, output_dir: Path
) -> None:
    model_file, default_input_size = SIZES[size]
    height = width = input_size or default_input_size
    precision_enum = PRECISIONS[precision]
    output_path = output_dir / f"fastsam-{size}" / "coreml" / f"fastsam-{size}_{precision}.pte"

    print(f"\n{'='*60}")
    print(f"Exporting fastsam-{size} to CoreML ({height}x{width}, {precision})")
    print(f"{'='*60}")

    print(f"Loading {model_file}...")
    inner = FastSAM(model_file).model

    print("Wrapping model...")
    model = FastSAMInstanceSegmentation(inner, method_size=height, sort_topk=True)
    model.eval()

    print(f"Tracing forward ({height}x{width})...")
    inputs = (torch.randn(1, 3, height, width),)
    with torch.no_grad():
        model(*inputs)

    print("Exporting to torch.export...")
    exported = export(model, inputs)

    print(f"Lowering to CoreML ({precision})...")
    compile_specs = CoreMLBackend.generate_compile_specs(compute_precision=precision_enum)
    et_prog = to_edge_transform_and_lower(
        {"forward": exported},
        partitioner=[
            CoreMLPartitioner(
                compile_specs=compile_specs,
                skip_ops_for_coreml_delegation=[],
            )
        ],
    ).to_executorch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_prog.buffer)

    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved {output_path}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FastSAM to CoreML via ExecutorTorch")
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
        help="Override input resolution (square). Defaults per variant: s=640, x=640",
    )
    parser.add_argument(
        "--precision",
        choices=list(PRECISIONS),
        default="fp32",
        help="Precision for CoreML export (default: fp32)",
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
            export_fastsam_coreml(size, args.input_size, args.precision, args.output_dir)
            successful += 1
        except Exception as exc:
            print(f"FAILED fastsam-{size} ({args.precision}): {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nDone: {successful} ok, {failed} failed")


if __name__ == "__main__":
    main()
