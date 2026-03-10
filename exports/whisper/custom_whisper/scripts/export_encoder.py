from custom_whisper import Whisper, load_model
from custom_whisper.model import AudioEncoder
from custom_whisper.preprocessor import ProcessorTorchModule
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from torch.export import Dim
from transformers import AutoProcessor
from typing import Optional
import argparse
import contextlib
import torch

# ----------------------
# Encoder module wrapper
# ----------------------

class EncoderWrapper(torch.nn.Module):
    def __init__(
        self, preprocessor: ProcessorTorchModule, encoder: torch.nn.Module
    ) -> None:
        super().__init__()
        self._preprocessor = preprocessor
        self._encoder = encoder
        self._n_fft = 512

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        log_mel = self._preprocessor(waveform)  # type: ignore
        assert log_mel.shape == torch.Size([80, 3000]), (
            "Input features have to be of size [80, 3000]!"
        )
        return self._encoder(log_mel.unsqueeze(0))

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
        # Load encoder (from openai-whisper)
        model = load_model(args.model_type)
        model.eval()
        encoder: AudioEncoder = model.encoder
        encoder.disable_sdpa()

        # Load preprocessor (from HF transformers)
        hf_preprocessor = AutoProcessor.from_pretrained(f"openai/whisper-{args.model_type}")
        preprocessor = ProcessorTorchModule(hf_preprocessor)
        preprocessor.eval()

        wrapped_encoder = EncoderWrapper(preprocessor, encoder)
    
    # -------------
    # Sample inputs
    # -------------

    SAMPLE_RATE = 16000
    inputs = (
      torch.randn(16 * SAMPLE_RATE, dtype=torch_dtype),
    )

    # Enable dynamic shapes
    T = Dim("T", min=201, max=30 * SAMPLE_RATE)
    dynamic_shapes = (
      {0: T},
    )

    exported_program = torch.export.export(wrapped_encoder, inputs, dynamic_shapes=dynamic_shapes)
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