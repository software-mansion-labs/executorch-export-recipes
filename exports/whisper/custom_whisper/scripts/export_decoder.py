from custom_whisper import Whisper, load_model
from custom_whisper.model import TextDecoder
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from torch.export import Dim
import argparse
import contextlib
import torch

# ----------------
# Context Managers
# ----------------

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

# -------------
# Main Function
# -------------

def main():

    # ----------------
    # Argument parsing
    # ----------------

    parser = argparse.ArgumentParser(description="Whisper Audio Transcription Tool")
    parser.add_argument("--model-type", type=str, default="tiny.en", help="Whisper model type (e.g., turbo, tiny.en)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help="Data type for model weights")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # -------------
    # Model loading
    # -------------

    print(f"\n[1/2] Loading model '{args.model_type}' with dtype {args.dtype}...")
    with _set_default_tensor_type(torch_dtype):
        model = load_model(args.model_type)
        model.eval()

        decoder: TextDecoder = model.decoder
    
    # -------------
    # Sample inputs
    # -------------

    inputs = (
        torch.randint(1, 100, size=(1, 64)),
        torch.arange(0, 64, dtype=torch.long),
        torch.randn(1, 1500, 384, dtype=torch_dtype)
    )

    # Enable dynamic shapes
    T = Dim("T", min=1, max=128)
    dynamic_shapes = (
        {1: T},
        {0: T},
        None
    )

    exported_program = torch.export.export(decoder, inputs, dynamic_shapes=dynamic_shapes)
    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner = [XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
        transform_passes=[AddmmToLinearTransform()],
    ).to_executorch()

    with open(args.output, "wb") as file:
        executorch_program.write_to_file(file)


# -----------
# Entry Point
# -----------

if __name__ == "__main__":
    main()