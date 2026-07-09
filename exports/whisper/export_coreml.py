# Export script: whisper_tiny.en -> CoreML (ExecuTorch bundled)
# Backend: CoreML | Precision: FLOAT32 | ComputeUnit: ALL
# Methods: encode + decode bundled into single .pte
# Date: 2026-05-07

import os
import sys
# scripts/profile.py shadows stdlib 'profile' — remove scripts/ from path before any imports
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = _scripts_dir
for _p in [_scripts_dir, ""]:
    while _p in sys.path:
        sys.path.remove(_p)
# Ensure the repo root is importable so custom_whisper resolves
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from custom_whisper import load_model
from custom_whisper.model import AudioEncoder, TextDecoder, LayerNorm
from custom_whisper.preprocessor import ProcessorTorchModule
import coremltools as ct
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.devtools.backend_debug.delegation_info import get_delegation_info
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from torch.export import Dim
from transformers import AutoProcessor
import argparse
import contextlib
import time
import torch


# ----------------------
# Bundled module wrappers (identical to export_bundled.py — do not modify)
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
    parser = argparse.ArgumentParser(description="Whisper CoreML Bundled Export")
    parser.add_argument("--model-type", type=str, default="tiny.en")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", type=str, required=True, help="Output .pte path")
    parser.add_argument("--int8-weights", action="store_true",
                        help="Enable CoreML int8 per-channel weight-only quantization")
    args = parser.parse_args()

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
        encoder.disable_sdpa()  # MANDATORY: disables SDPA globally (affects decoder too)

        decoder: TextDecoder = model.decoder

        # LayerNorm.forward upcasts the activation to fp32 (x.float()) but leaves gamma/beta
        # in the model dtype. For fp16/bf16 exports this yields a layer_norm op with fp32 x
        # and fp16 gamma, which CoreML rejects. Keep gamma/beta in fp32 to match the activation;
        # the output is cast back to x.dtype inside forward, so this is safe (no-op for fp32).
        for module in model.modules():
            if isinstance(module, LayerNorm):
                module.float()

    # The preprocessor MUST be built outside the fp16/bf16 default-dtype context. Its STFT
    # kernel construction computes an integer outer product k * n (up to 256 * 511 = 130816)
    # which overflows the fp16 max (65504) -> inf -> cos(inf) = NaN, poisoning the whole encoder.
    # Audio feature extraction also wants fp32 precision; the EncoderWrapper casts the resulting
    # log-mel to the encoder dtype, and the runtime always feeds a fp32 waveform.
    hf_preprocessor = AutoProcessor.from_pretrained(f"openai/whisper-{args.model_type}")
    preprocessor = ProcessorTorchModule(hf_preprocessor)
    preprocessor.eval()

    encoder_wrapper = EncoderWrapper(preprocessor, encoder)
    decoder_wrapper = DecoderWrapper(decoder, dtype=torch_dtype)

    # ------------------
    # Exporting methods
    # ------------------
    print(f"[2/3] Exporting 'encode' and 'decode' methods...")

    SAMPLE_RATE = 16000
    # Waveform is always fp32 (matches the runtime, which loads audio as fp32); the preprocessor
    # runs in fp32 and the EncoderWrapper casts the log-mel to the encoder dtype internally.
    encode_inputs = (torch.randn(16 * SAMPLE_RATE, dtype=torch.float32),)
    T_audio = Dim("T_audio", min=201, max=30 * SAMPLE_RATE)
    encode_dynamic_shapes = ({0: T_audio},)

    print("  - Exporting 'encode'...")
    exported_encode = torch.export.export(
        encoder_wrapper, encode_inputs, dynamic_shapes=encode_dynamic_shapes, strict=False
    )

    n_audio_state = model.dims.n_audio_state
    decode_inputs = (
        torch.randint(1, 100, size=(1, 1)),
        torch.tensor([0], dtype=torch.long),
        torch.randn(1, 1500, n_audio_state, dtype=torch.float32),
    )

    print("  - Exporting 'decode'...")
    exported_decode = torch.export.export(
        decoder_wrapper, decode_inputs, strict=False
    )

    # 3. Export the dynamic-dim companion methods (no args, constant outputs)
    print("  - Exporting 'get_dynamic_dims_encode' / 'get_dynamic_dims_decode'...")
    exported_get_dynamic_dims_encode = torch.export.export(
        build_dynamic_dims_module(encode_inputs, encode_dynamic_shapes), ()
    )
    exported_get_dynamic_dims_decode = torch.export.export(
        build_dynamic_dims_module(decode_inputs, (None, None, None)), ()
    )

    # ----------------
    # CoreML compile specs
    # ----------------
    print(f"[3/3] Lowering to ExecuTorch CoreML and saving to '{args.output}'...")
    t0 = time.time()

    compile_spec_kwargs = dict(
        compute_precision=ct.precision.FLOAT32,   # FLOAT16 hallucinates on Whisper
        compute_unit=ct.ComputeUnit.ALL,            # NE+GPU+CPU; transformer benefits from GPU
        model_type=CoreMLBackend.MODEL_TYPE.MODEL,
        minimum_deployment_target=ct.target.iOS16
    )
    if args.int8_weights:
        compile_spec_kwargs["op_linear_quantizer_config"] = {
            "mode": "linear_symmetric",
            "dtype": "int8",
            "granularity": "per_channel",
            "weight_threshold": 2048,
        }

    compile_specs = CoreMLBackend.generate_compile_specs(**compile_spec_kwargs)

    encode_partitioner = CoreMLPartitioner(compile_specs=compile_specs, take_over_mutable_buffer=False)
    decode_partitioner = CoreMLPartitioner(compile_specs=compile_specs, take_over_mutable_buffer=False)

    edge_manager = to_edge_transform_and_lower(
        {"encode": exported_encode, 
         "decode": exported_decode,
         "get_dynamic_dims_encode": exported_get_dynamic_dims_encode,
         "get_dynamic_dims_decode": exported_get_dynamic_dims_decode,
        },
        partitioner={
            "encode": [encode_partitioner],
            "decode": [decode_partitioner],
        },
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
        transform_passes=[AddmmToLinearTransform()],
    )

    # Print delegation stats per method
    for method_name in ["encode", "decode"]:
        ep = edge_manager.exported_program(method_name)
        info = get_delegation_info(ep.graph_module)
        delegated = info.num_delegated_nodes
        total = info.num_delegated_nodes + info.num_non_delegated_nodes
        print(f"DELEGATION {method_name}: delegated {delegated} / {total} nodes")

    executorch_program = edge_manager.to_executorch()

    elapsed = time.time() - t0
    print(f"Export completed in {elapsed:.1f}s")

    with open(args.output, "wb") as file:
        executorch_program.write_to_file(file)

    print(f"Success! Bundled CoreML model saved to '{args.output}'.")


if __name__ == "__main__":
    main()
