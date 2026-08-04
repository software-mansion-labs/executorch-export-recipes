import contextlib
import importlib.util
import sys
from pathlib import Path

import torch

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_custom_whisper():
    pkg_init = _MODEL_DIR / "_custom_whisper" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "whisper_coreml_custom_whisper", pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_custom_whisper")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["whisper_coreml_custom_whisper"] = mod
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
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from torch.export import Dim
    from transformers import AutoProcessor

    cw = _load_custom_whisper()
    preprocessor_mod = importlib.import_module("whisper_coreml_custom_whisper.preprocessor")
    model_mod = importlib.import_module("whisper_coreml_custom_whisper.model")
    ProcessorTorchModule = preprocessor_mod.ProcessorTorchModule
    LayerNorm = model_mod.LayerNorm

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

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, preprocessor, encoder):
            super().__init__()
            self._preprocessor = preprocessor
            self._encoder = encoder

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            log_mel = self._preprocessor(waveform)
            return self._encoder(log_mel.unsqueeze(0))

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, tokens: torch.LongTensor, positions: torch.LongTensor, encodings: torch.FloatTensor):
            return self.decoder(tokens, positions, encodings)

    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        config["variant"]["precision"]
    ]
    size = config.get("size") or "tiny"
    torch_dtype = torch.float32

    with _set_default_tensor_type(torch_dtype):
        model = cw.load_model(size)
        model.eval()

        encoder = model.encoder
        encoder.disable_sdpa()

        decoder = model.decoder
        if hasattr(decoder, "embed_positions"):
            decoder.embed_positions = WhisperPositionalEmbedding(
                decoder.embed_positions.weight
            )

        # LayerNorm.forward upcasts the activation to fp32 (x.float()) but leaves gamma/beta
        # in the model dtype. For fp16 exports this yields a layer_norm op with fp32 x and
        # fp16 gamma, which CoreML rejects. Keep gamma/beta in fp32 to match the activation.
        for module in model.modules():
            if isinstance(module, LayerNorm):
                module.float()

    # The preprocessor MUST be built outside the fp16/bf16 default-dtype context. Its STFT
    # kernel construction computes an integer outer product k * n (up to 256 * 511 = 130816)
    # which overflows fp16 max (65504) -> inf -> cos(inf) = NaN, poisoning the whole encoder.
    hf_preprocessor = AutoProcessor.from_pretrained(f"openai/whisper-{size}")
    preprocessor = ProcessorTorchModule(hf_preprocessor)
    preprocessor.eval()

    encoder_wrapper = EncoderWrapper(preprocessor, encoder)
    decoder_wrapper = DecoderWrapper(decoder)

    SAMPLE_RATE = 16000
    encode_inputs = (torch.randn(16 * SAMPLE_RATE, dtype=torch_dtype),)
    T_audio = Dim("T_audio", min=201, max=30 * SAMPLE_RATE)
    encode_dynamic_shapes = ({0: T_audio},)

    exported_encode = torch.export.export(
        encoder_wrapper, encode_inputs, dynamic_shapes=encode_dynamic_shapes, strict=False
    )

    n_audio_state = model.dims.n_audio_state
    # The shipped CoreML decode is static single-token (tokens [1,1], pos [1]),
    # unlike the XNNPACK dynamic [1,128] decode.
    decode_inputs = (
        torch.randint(1, 100, size=(1, 1)),
        torch.tensor([0], dtype=torch.long),
        torch.randn(1, 1500, n_audio_state, dtype=torch_dtype),
    )

    exported_decode = torch.export.export(decoder_wrapper, decode_inputs, strict=False)

    n_vocab = model.dims.n_vocab
    schema_json = load_and_validate_schema(
        _MODEL_DIR / "schema.json",
        replacements={"n_audio_state": str(n_audio_state), "n_vocab": str(n_vocab)},
    )

    methods = {
        "encode": exported_encode,
        "decode": exported_decode,
    }

    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )

    # take_over_mutable_buffer=False: critical for Whisper — the KV cache buffers are updated
    # via in-place scatter ops (self.k_cache[positions] = ...). If CoreML takes ownership of
    # those buffers it tries to lower the scatter as a MIL in-place op, which is invalid and
    # produces an InvalidProgram error at runtime. Leaving them as CPU buffers avoids this.
    encode_partitioner = CoreMLPartitioner(compile_specs=specs, take_over_mutable_buffer=False)
    decode_partitioner = CoreMLPartitioner(compile_specs=specs, take_over_mutable_buffer=False)

    executorch_program = to_edge_transform_and_lower(
        methods,
        partitioner={"encode": [encode_partitioner], "decode": [decode_partitioner]},
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
        transform_passes=[AddmmToLinearTransform()],
        constant_methods={"get_model_schema": schema_json},
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(executorch_program.buffer)
    return str(out_path)
