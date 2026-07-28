import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_custom_whisper():
    pkg_init = _MODEL_DIR / "_custom_whisper" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "whisper_tiny_mlx_custom_whisper",
        pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_custom_whisper")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["whisper_tiny_mlx_custom_whisper"] = mod
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(torch.float)


def export(config: dict, out_dir: str) -> str:
    import sys

    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import Dim
    from transformers import AutoProcessor

    cw = _load_custom_whisper()
    preprocessor_mod = importlib.import_module("whisper_tiny_mlx_custom_whisper.preprocessor")
    ProcessorTorchModule = preprocessor_mod.ProcessorTorchModule
    model_mlx_mod = importlib.import_module("whisper_tiny_mlx_custom_whisper.model_mlx")
    TextDecoderMlx = model_mlx_mod.TextDecoderMlx

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, preprocessor, encoder):
            super().__init__()
            self._preprocessor = preprocessor
            self._encoder = encoder
            self._n_fft = 512
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

    precision = config["variant"].get("precision", "fp32")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32, "fp16": torch.float16}
    torch_dtype = dtype_map[precision]
    size = config.get("size") or "tiny"

    with _set_default_tensor_type(torch_dtype):
        model = cw.load_model(size)
        model.eval()

        encoder = model.encoder
        encoder.disable_sdpa()

        decoder = model.decoder
        for module in model.modules():
            if isinstance(module, cw.model.LayerNorm):
                module.float()

        hf_preprocessor = AutoProcessor.from_pretrained(f"openai/whisper-{size}")
        preprocessor = ProcessorTorchModule(hf_preprocessor)
        preprocessor.eval()

        encoder_wrapper = EncoderWrapper(preprocessor, encoder)
        decode_module = TextDecoderMlx.from_decoder(
            decoder, max_cache_len=224, dtype=torch_dtype
        ).eval()
        decoder_wrapper = DecoderWrapper(decode_module, dtype=torch_dtype)

    SAMPLE_RATE = 16000
    encode_inputs = (torch.randn(16 * SAMPLE_RATE, dtype=torch.float32),)
    T_audio = Dim("T_audio", min=201, max=30 * SAMPLE_RATE)
    encode_dynamic_shapes = ({0: T_audio},)

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

    exported_decode = torch.export.export(
        decoder_wrapper, decode_inputs, strict=False
    )
    exported_decode = exported_decode.run_decompositions({})

    n_vocab = model.dims.n_vocab
    schema_template = (_MODEL_DIR / "schema.json").read_text()
    schema_json = schema_template.replace("{n_audio_state}", str(n_audio_state)).replace("{n_vocab}", str(n_vocab))

    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )

    methods = {
        "encode": exported_encode,
        "decode": exported_decode,
    }

    edge_manager = to_edge_transform_and_lower(
        methods,
        partitioner={
            "encode": [MLXPartitioner()],
            "decode": [MLXPartitioner()],
        },
        transform_passes=get_default_passes(),
        compile_config=edge_config,
        constant_methods={"get_model_schema": schema_json},
    )

    from executorch.exir.capture._config import ExecutorchBackendConfig
    executorch_program = edge_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as file:
        executorch_program.write_to_file(file)

    return str(out_path)
