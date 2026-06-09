# Export script: whisper_tiny.en -> CoreML (ExecuTorch bundled)
# Backend: CoreML | Precision: FLOAT32 | ComputeUnit: ALL
# Methods: encode + decode bundled into single .pte
# Date: 2026-05-07

import os
import sys
# scripts/profile.py shadows stdlib 'profile' — remove scripts/ from path before any imports
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_scripts_dir)
for _p in [_scripts_dir, ""]:
    while _p in sys.path:
        sys.path.remove(_p)
# Ensure the repo root is importable so custom_whisper resolves
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from custom_whisper import load_model
from custom_whisper.model import AudioEncoder, TextDecoder
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

        hf_preprocessor = AutoProcessor.from_pretrained(f"openai/whisper-{args.model_type}")
        preprocessor = ProcessorTorchModule(hf_preprocessor)
        preprocessor.eval()

        encoder_wrapper = EncoderWrapper(preprocessor, encoder)
        decoder_wrapper = DecoderWrapper(decoder)

    # ------------------
    # Exporting methods
    # ------------------
    print(f"[2/3] Exporting 'encode' and 'decode' methods...")

    SAMPLE_RATE = 16000
    encode_inputs = (torch.randn(16 * SAMPLE_RATE, dtype=torch_dtype),)
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
        torch.randn(1, 1500, n_audio_state, dtype=torch_dtype),
    )

    print("  - Exporting 'decode'...")
    exported_decode = torch.export.export(
        decoder_wrapper, decode_inputs, strict=False
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
        {"encode": exported_encode, "decode": exported_decode},
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
