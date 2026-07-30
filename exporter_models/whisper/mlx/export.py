import contextlib
import importlib
import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# Self-attention KV cache length for the MLX decoder.
MAX_SEQ_LEN = 128


def _load_custom_whisper():
    pkg_init = _MODEL_DIR / "_custom_whisper" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "whisper_mlx_custom_whisper", pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_custom_whisper")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["whisper_mlx_custom_whisper"] = mod
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(torch.float)


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


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from torch.export import Dim
    from transformers import AutoProcessor

    cw = _load_custom_whisper()
    preprocessor_mod = importlib.import_module("whisper_mlx_custom_whisper.preprocessor")
    ProcessorTorchModule = preprocessor_mod.ProcessorTorchModule
    model_mlx_mod = importlib.import_module("whisper_mlx_custom_whisper.model_mlx")
    TextDecoderMlx = model_mlx_mod.TextDecoderMlx

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, preprocessor, encoder):
            super().__init__()
            self._preprocessor = preprocessor
            self._encoder = encoder
            self._encoder_dtype = next(encoder.parameters()).dtype

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            log_mel = self._preprocessor(waveform)
            return self._encoder(log_mel.unsqueeze(0).to(self._encoder_dtype)).float()

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder, dtype: torch.dtype):
            super().__init__()
            self.decoder = decoder
            self.dtype = dtype

        def forward(self, tokens: torch.LongTensor, positions: torch.LongTensor, encodings: torch.FloatTensor):
            return self.decoder(tokens, positions, encodings.to(self.dtype))

    size = config.get("size") or "tiny"
    torch_dtype = torch.float32

    with _set_default_tensor_type(torch_dtype):
        model = cw.load_model(size)
        model.eval()

        encoder = model.encoder
        # main's MLX path keeps encoder SDPA enabled (do NOT disable it).

        decoder = model.decoder

        # Keep LayerNorm gamma/beta in fp32 to match the upcast activation inside
        # LayerNorm.forward (no-op for float32).
        LayerNorm = cw.model.LayerNorm
        for module in model.modules():
            if isinstance(module, LayerNorm):
                module.float()

        hf_preprocessor = AutoProcessor.from_pretrained(f"openai/whisper-{size}")
        preprocessor = ProcessorTorchModule(hf_preprocessor)
        preprocessor.eval()

        encoder_wrapper = EncoderWrapper(preprocessor, encoder)

        decode_module = TextDecoderMlx.from_decoder(
            decoder, max_cache_len=MAX_SEQ_LEN, dtype=torch_dtype
        ).eval()
        decoder_wrapper = DecoderWrapper(decode_module, dtype=torch_dtype)

    SAMPLE_RATE = 16000
    encode_inputs = (torch.randn(16 * SAMPLE_RATE, dtype=torch_dtype),)
    T_audio = Dim("T_audio", min=201, max=30 * SAMPLE_RATE)
    encode_dynamic_shapes = ({0: T_audio},)

    exported_encode = torch.export.export(
        encoder_wrapper, encode_inputs, dynamic_shapes=encode_dynamic_shapes, strict=False
    )
    exported_encode = exported_encode.run_decompositions({})

    n_audio_state = model.dims.n_audio_state
    # The MLX decoder runs one token at a time against a fixed-length self-attention
    # KV cache, so every 'decode' input axis is static (no dynamic_shapes).
    decode_inputs = (
        torch.randint(1, 100, size=(1, 1)),
        torch.tensor([0], dtype=torch.long),
        torch.randn(1, 1500, n_audio_state, dtype=torch_dtype),
    )
    decode_dynamic_shapes = (None, None, None)

    exported_decode = torch.export.export(
        decoder_wrapper, decode_inputs, strict=False
    )
    exported_decode = exported_decode.run_decompositions({})

    # Export the dynamic-dim companion methods (no args, constant outputs).
    exported_get_dynamic_dims_encode = torch.export.export(
        build_dynamic_dims_module(encode_inputs, encode_dynamic_shapes), ()
    )
    exported_get_dynamic_dims_decode = torch.export.export(
        build_dynamic_dims_module(decode_inputs, decode_dynamic_shapes), ()
    )

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

    executorch_program = edge_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(executorch_program.buffer)
    return str(out_path)
