# Export script: whisper tiny.en -> MLX (ExecuTorch bundled)
# Backend: MLX (Apple Silicon GPU via MLX) | Precision: FLOAT32
# Methods: encode + decode bundled into single .pte
#
# This mirrors export_coreml.py but swaps the CoreML partitioner for the MLX
# delegate partitioner (executorch.backends.mlx.MLXPartitioner). The encoder
# and preprocessor are reused from custom_whisper as-is; the decoder is rebuilt
# from custom_whisper.model_mlx.TextDecoderMlx, whose self-attention uses the
# MLX delegate's KV-cache primitives (mlx::kv_cache_update + mlx::custom_sdpa)
# so the cache writes and attention fully delegate to MLX instead of falling back
# to the portable CPU runtime. The I/O contract stays compatible with multi_test.py.

import os
import sys
# custom_whisper/ lives next to this script; make sure it is importable regardless
# of the working directory the script is launched from.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from custom_whisper import load_model
from custom_whisper.model import AudioEncoder, TextDecoder, LayerNorm
from custom_whisper.model_mlx import TextDecoderMlx, MAX_SEQ_LEN as DEFAULT_CACHE_LEN
from custom_whisper.preprocessor import ProcessorTorchModule
from executorch.backends.mlx import MLXPartitioner
from executorch.backends.mlx.passes import get_default_passes
from executorch.devtools.backend_debug.delegation_info import get_delegation_info
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from torch.export import Dim
from transformers import AutoProcessor
import argparse
import contextlib
import time
import torch


# ----------------------
# Bundled module wrappers (mirror export_coreml.py)
# ----------------------

class EncoderWrapper(torch.nn.Module):
    def __init__(
        self, preprocessor: ProcessorTorchModule, encoder: torch.nn.Module
    ) -> None:
        super().__init__()
        self._preprocessor = preprocessor
        self._encoder = encoder
        self._n_fft = 512
        self._encoder_dtype = next(encoder.parameters()).dtype

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        log_mel = self._preprocessor(waveform)
        return self._encoder(log_mel.unsqueeze(0).to(self._encoder_dtype)).float()


class DecoderWrapper(torch.nn.Module):
    def __init__(
        self, decoder: TextDecoder, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.dtype = dtype

    def forward(self, tokens: torch.LongTensor, positions: torch.LongTensor, encodings: torch.FloatTensor):
        return self.decoder(tokens, positions, encodings.to(self.dtype))


# ---------------------------------
# Dynamic-shape companion methods
# ---------------------------------

# The RNE runtime cannot recover the active (min, max, step) range of a dynamic
# input dimension from .pte metadata alone: ExecuTorch only serializes the static
# upper bound, not the live symbolic range. To expose it, we bundle a companion
# method "get_dynamic_dims_<method>" next to each exported method. It takes no
# arguments and returns, for every Tag::Tensor input of <method>, a [rank, 3]
# int32 tensor whose rows are [min, max, step] per axis (static axes use
# [size, size, 1]). See model.cpp::parseDynamicInputShapes for the contract.

class GetDynamicDims(torch.nn.Module):
    def __init__(self, rows_per_input) -> None:
        super().__init__()
        self._rows_per_input = rows_per_input

    def forward(self):
        return tuple(
            torch.tensor(rows, dtype=torch.int32) for rows in self._rows_per_input
        )


def build_dynamic_dims_module(example_inputs, dynamic_shapes) -> GetDynamicDims:
    """Derive a GetDynamicDims companion module from a method's example inputs and
    its torch.export `dynamic_shapes` spec, keeping the reported ranges in
    lock-step with what was actually exported."""
    rows_per_input = []
    for example, axis_map in zip(example_inputs, dynamic_shapes):
        axis_map = axis_map or {}
        rows = []
        for axis, size in enumerate(example.shape):
            dim = axis_map.get(axis)
            if dim is None:
                rows.append([int(size), int(size), 1])  # static axis
            else:
                rows.append([int(dim.min), int(dim.max), 1])
        rows_per_input.append(rows)
    return GetDynamicDims(rows_per_input)


# ----------------
# Context Managers
# ----------------

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


# -------------
# Main Function
# -------------

def main():
    parser = argparse.ArgumentParser(description="Whisper MLX Bundled Export")
    parser.add_argument("--model-type", type=str, default="tiny.en")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", type=str, required=True, help="Output .pte path")
    parser.add_argument(
        "--no-encoder-sdpa", action="store_true",
        help="Disable SDPA on the encoder (default: encoder SDPA enabled). The MLX "
             "decoder always uses mlx::custom_sdpa.",
    )
    parser.add_argument(
        "--legacy-decoder", action="store_true",
        help="Use the original buffer-based TextDecoder (KV cache via aten.index_put, "
             "not delegated) instead of the MLX TextDecoderMlx. Useful for A/B comparison.",
    )
    parser.add_argument(
        "--max-cache-len", type=int, default=DEFAULT_CACHE_LEN,
        help=f"Self-attention KV cache length (default {DEFAULT_CACHE_LEN})",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Enable ET MLX debug logging (ET_MLX_DEBUG=1)")
    args = parser.parse_args()

    if args.debug:
        os.environ["ET_MLX_DEBUG"] = "1"

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # -------------
    # Model loading
    # -------------
    print(f"\n[1/3] Loading model '{args.model_type}' with dtype {args.dtype}...")
    with _set_default_tensor_type(torch_dtype):
        model = load_model(args.model_type)
        model.eval()

        encoder: AudioEncoder = model.encoder
        if args.no_encoder_sdpa:
            encoder.disable_sdpa()  # default: SDPA enabled on the encoder

        decoder: TextDecoder = model.decoder

        # Keep LayerNorm gamma/beta in fp32 to match the upcast activation inside
        # LayerNorm.forward (no-op for float32).
        for module in model.modules():
            if isinstance(module, LayerNorm):
                module.float()

    # The preprocessor MUST be built outside the fp16/bf16 default-dtype context
    # (its STFT kernel construction overflows fp16). For fp32 this is harmless.
    hf_preprocessor = AutoProcessor.from_pretrained(f"openai/whisper-{args.model_type}")
    preprocessor = ProcessorTorchModule(hf_preprocessor)
    preprocessor.eval()

    encoder_wrapper = EncoderWrapper(preprocessor, encoder)

    if args.legacy_decoder:
        print("  - Using legacy buffer-based TextDecoder (KV cache NOT delegated)")
        decode_module = decoder
    else:
        print("  - Using MLX TextDecoderMlx (KV cache + custom_sdpa delegated)")
        decode_module = TextDecoderMlx.from_decoder(
            decoder, max_cache_len=args.max_cache_len, dtype=torch_dtype
        ).eval()
    decoder_wrapper = DecoderWrapper(decode_module, dtype=torch_dtype)

    # ------------------
    # Exporting methods
    # ------------------
    print(f"[2/3] Exporting 'encode' and 'decode' methods...")

    SAMPLE_RATE = 16000
    encode_inputs = (torch.randn(16 * SAMPLE_RATE, dtype=torch.float32),)
    T_audio = Dim("T_audio", min=201, max=30 * SAMPLE_RATE)
    encode_dynamic_shapes = ({0: T_audio},)

    print("  - Exporting 'encode'...")
    exported_encode = torch.export.export(
        encoder_wrapper, encode_inputs, dynamic_shapes=encode_dynamic_shapes, strict=False
    )
    exported_encode = exported_encode.run_decompositions({})

    n_audio_state = model.dims.n_audio_state
    decode_inputs = (
        torch.randint(1, 100, size=(1, 1)),
        torch.tensor([0], dtype=torch.long),
        torch.randn(1, 1500, n_audio_state, dtype=torch.float32),
    )
    # The MLX decoder runs one token at a time against a fixed-length self-attention
    # KV cache, so every 'decode' input axis is static (no dynamic_shapes).
    decode_dynamic_shapes = (None, None, None)

    print("  - Exporting 'decode'...")
    exported_decode = torch.export.export(
        decoder_wrapper, decode_inputs, strict=False
    )
    exported_decode = exported_decode.run_decompositions({})

    # Export the dynamic-dim companion methods (no args, constant outputs).
    print("  - Exporting 'get_dynamic_dims_encode' / 'get_dynamic_dims_decode'...")
    exported_get_dynamic_dims_encode = torch.export.export(
        build_dynamic_dims_module(encode_inputs, encode_dynamic_shapes), ()
    )
    exported_get_dynamic_dims_decode = torch.export.export(
        build_dynamic_dims_module(decode_inputs, decode_dynamic_shapes), ()
    )

    # ----------------
    # Lower to ExecuTorch with MLX delegate
    # ----------------
    print(f"[3/3] Lowering to ExecuTorch MLX and saving to '{args.output}'...")
    t0 = time.time()

    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )

    edge_manager = to_edge_transform_and_lower(
        {
            "encode": exported_encode,
            "decode": exported_decode,
            "get_dynamic_dims_encode": exported_get_dynamic_dims_encode,
            "get_dynamic_dims_decode": exported_get_dynamic_dims_decode,
        },
        partitioner={
            # Only the real compute methods are delegated to MLX; the companion
            # methods just return constant int32 tensors and stay portable.
            "encode": [MLXPartitioner()],
            "decode": [MLXPartitioner()],
        },
        transform_passes=get_default_passes(),
        compile_config=edge_config,
    )

    # Print delegation stats per method
    for method_name in ["encode", "decode"]:
        ep = edge_manager.exported_program(method_name)
        info = get_delegation_info(ep.graph_module)
        delegated = info.num_delegated_nodes
        total = info.num_delegated_nodes + info.num_non_delegated_nodes
        print(f"DELEGATION {method_name}: delegated {delegated} / {total} nodes")

    from executorch.exir.capture._config import ExecutorchBackendConfig
    executorch_program = edge_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    elapsed = time.time() - t0
    print(f"Export completed in {elapsed:.1f}s")

    with open(args.output, "wb") as file:
        executorch_program.write_to_file(file)

    print(f"Success! Bundled MLX model saved to '{args.output}'.")


if __name__ == "__main__":
    main()
