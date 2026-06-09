from custom_whisper import load_model
from custom_whisper.model import AudioEncoder, TextDecoder
from custom_whisper.preprocessor import ProcessorTorchModule
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from torch.export import Dim
from torchao.quantization import MappingType, quantize_
from torchao.utils import unwrap_tensor_subclass
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from transformers import AutoProcessor
import argparse
import contextlib
import torch

# ----------------------
# Quantization Helpers
# ----------------------

# WARNING:
# This is used because HF implementation of Whisper's PE inherits from nn.Embedding, which causes
# the embedding_transform to treat is as a normal embededing layer, causing issues at runtime.
# This is the HF implementation copied, but it doesn't inherit from nn.module. Then the PE layer within
# the decoder is replaced with the instance of the class below.
class WhisperPositionalEmbedding(torch.nn.Module):
    def __init__(self, pretrained_weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(pretrained_weight.clone().detach())

    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            return self.weight[
                past_key_values_length : past_key_values_length + input_ids.shape[1]
            ]
        else:
            return self.weight[position_ids]

def torchao_quantize(model: torch.nn.Module, group_size: int = 64, quantize_embeddings: bool = False, embedding_config: str = "8,64"):
    print(f"Running torchao quantization for: {model.__class__.__name__} with group_size {group_size}")
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig

    if quantize_embeddings:
        # Check if model or its components have decoder with embeddings
        # The wrappers might have different structures, but we target the components
        # For Whisper, we target the TextDecoder inside the wrapper
        decoder = None
        if hasattr(model, "decoder") and isinstance(model.decoder, torch.nn.Module):
            decoder = model.decoder
        elif hasattr(model, "decoder_wrapper") and hasattr(model.decoder_wrapper, "decoder"):
             decoder = model.decoder_wrapper.decoder
        
        if decoder is not None:
            print(f"  - Quantizing embeddings with config: {embedding_config}")
            # Apply WhisperPositionalEmbedding hack to decoder
            if hasattr(decoder, "embed_positions"):
                decoder.embed_positions = WhisperPositionalEmbedding(
                    decoder.embed_positions.weight
                )
            
            # Apply embedding quantization
            embedding_transforms = get_quant_embedding_transform(
                embedding_config, use_shared_embedding=True
            )
            # Find the decoder within the model structure and transform it
            # If the model is the decoder itself, transform it
            if isinstance(model, TextDecoder) or "TextDecoder" in model.__class__.__name__:
                model = embedding_transforms(model)
            elif hasattr(model, "decoder"):
                model.decoder = embedding_transforms(model.decoder)

    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(group_size),
            weight_mapping_type=MappingType.SYMMETRIC,
        ),
    )
    return unwrap_tensor_subclass(model)

# ----------------------
# Bundled module wrapper
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
        log_mel = self._preprocessor(waveform)
        return self._encoder(log_mel.unsqueeze(0))


class DecoderWrapper(torch.nn.Module):
    def __init__(
        self, decoder: TextDecoder
    ) -> None:
      super().__init__()

      self.decoder = decoder

    def forward(self, tokens: torch.LongTensor, positions: torch.LongTensor, encodings: torch.FloatTensor):
      return self.decoder(tokens, positions, encodings)

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

    parser = argparse.ArgumentParser(description="Whisper Bundled Export Tool")
    parser.add_argument("--model-type", type=str, default="tiny.en", help="Whisper model type (e.g., turbo, tiny.en)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help="Data type for model weights")
    parser.add_argument("--quantize", action="store_true", help="Whether to quantize the model using torchao (int4 weight-only)")
    parser.add_argument("--group-size", type=int, default=64, help="Group size for quantization (default: 64)")
    parser.add_argument("--quantize-embeddings", action="store_true", help="Whether to quantize embeddings")
    parser.add_argument("--embedding-config", type=str, default="8,64", help="Embedding quantization config (default: '8,64')")
    parser.add_argument("--output", type=str, required=True, help="Output file path (.pte)")
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

    print(f"\n[1/3] Loading model '{args.model_type}' with dtype {args.dtype}...")
    with _set_default_tensor_type(torch_dtype):
        # Load from custom_whisper
        model = load_model(args.model_type)
        model.eval()
        
        # Load components
        encoder: AudioEncoder = model.encoder
        encoder.disable_sdpa() # This sets MultiHeadAttention.use_sdpa = False globally
        
        decoder: TextDecoder = model.decoder

        # Load preprocessor (from HF transformers)
        hf_preprocessor = AutoProcessor.from_pretrained(f"openai/whisper-{args.model_type}")
        preprocessor = ProcessorTorchModule(hf_preprocessor)
        preprocessor.eval()

        encoder_wrapper = EncoderWrapper(preprocessor, encoder)
        decoder_wrapper = DecoderWrapper(decoder)

        if args.quantize:
            print(f"[1.1/3] Applying quantization (group_size={args.group_size}, quantize_embeddings={args.quantize_embeddings})...")
            encoder_wrapper = torchao_quantize(
                encoder_wrapper,
                group_size=args.group_size,
                quantize_embeddings=False # Encoder doesn't have text embeddings
            )
            decoder_wrapper = torchao_quantize(
                decoder_wrapper,
                group_size=args.group_size,
                quantize_embeddings=args.quantize_embeddings,
                embedding_config=args.embedding_config
            )

    # ------------------
    # Exporting methods
    # ------------------

    print(f"[2/3] Exporting 'encode' and 'decode' methods...")

    # 1. Export ENCODE
    SAMPLE_RATE = 16000
    encode_inputs = (
      torch.randn(16 * SAMPLE_RATE, dtype=torch_dtype),
    )
    # Enable dynamic shapes for encode
    T_audio = Dim("T_audio", min=201, max=30 * SAMPLE_RATE)
    encode_dynamic_shapes = ({0: T_audio},)
    
    print("  - Exporting 'encode'...")
    exported_encode = torch.export.export(encoder_wrapper, encode_inputs, dynamic_shapes=encode_dynamic_shapes)

    # 2. Export DECODE
    # tokens: (1, 64), positions: (64,), encodings: (1, 1500, n_audio_state)
    n_audio_state = model.dims.n_audio_state
    decode_inputs = (
        torch.randint(1, 100, size=(1, 64)),
        torch.arange(0, 64, dtype=torch.long),
        torch.randn(1, 1500, n_audio_state, dtype=torch_dtype)
    )
    # Enable dynamic shapes for decode
    T_text = Dim("T_text", min=1, max=128)
    decode_dynamic_shapes = (
        {1: T_text},
        {0: T_text},
        None
    )
    
    print("  - Exporting 'decode'...")
    exported_decode = torch.export.export(decoder_wrapper, decode_inputs, dynamic_shapes=decode_dynamic_shapes)

    # ----------------
    # Combining to ET
    # ----------------

    print(f"[3/3] Lowering to ExecuTorch and saving to '{args.output}'...")
    
    # Bundle into a dictionary for to_edge_transform_and_lower
    methods = {
        "encode": exported_encode,
        "decode": exported_decode,
    }

    # Apply lowering and partition to XNNPACK
    executorch_program = to_edge_transform_and_lower(
        methods,
        partitioner = [XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
        transform_passes=[AddmmToLinearTransform()],
    ).to_executorch()

    # Write the bundled program to file
    with open(args.output, "wb") as file:
        executorch_program.write_to_file(file)

    print("Success! Bundled model exported.")

# -----------
# Entry Point
# -----------

if __name__ == "__main__":
    main()
