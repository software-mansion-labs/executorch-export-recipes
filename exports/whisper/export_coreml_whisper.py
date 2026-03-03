import time

from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
from datasets import load_dataset
from torchao.quantization import MappingType, quantize_
from torchao.utils import unwrap_tensor_subclass


from examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.runtime import Program, Runtime
from executorch.exir.capture._config import EdgeCompileConfig
from executorch.exir.program._program import to_edge_transform_and_lower
import torchaudio
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    GenerationConfig,
)
from torch.export import Dim
import lovely_tensors as lt  # type: ignore

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
import coremltools as ct
import logging


logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)
lt.monkey_patch()

MODEL_NAME = "openai/whisper-tiny"
ENCODER_OUT_FILENAME_GPU = "whisper_encoder_coreml_cpu_and_gpu.pte"
ENCODER_OUT_FILENAME_ALL = "whisper_encoder_coreml_all.pte"
DECODER_OUT_FILENAME_GPU = "whisper_decoder_coreml_cpu_and_gpu.pte"
DECODER_OUT_FILENAME_ALL = "whisper_decoder_coreml_all.pte"
SR_INT = int(16e3)
MAX_SEQ_LEN = 128
GENERATION_CONFIG = None


def torchao_quantize(model: torch.nn.Module, group_size: int):
    logger.info(f"Running torchao quantization for: {model.__class__}")
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig

    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,  # type: ignore
            weight_granularity=PerGroup(group_size),
            weight_mapping_type=MappingType.SYMMETRIC,
        ),
    )  # type: ignore
    return unwrap_tensor_subclass(model)


class AudioUtils:
    @staticmethod
    def load_audio_at_sr(
        audio: np.ndarray, orig_sr: int, target_sr: int = SR_INT
    ) -> torch.Tensor:
        waveform = torch.from_numpy(audio)
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=target_sr
            )
            waveform = resampler(waveform)
        return waveform


class ProcessorTorchModule(torch.nn.Module):
    def __init__(self, processor: AutoProcessor):
        super().__init__()
        self.feature_extractor = processor.feature_extractor  # type: ignore
        self.n_fft = self.feature_extractor.n_fft
        self.hop_length = self.feature_extractor.hop_length
        self.n_mels = self.feature_extractor.feature_size
        self.sampling_rate = self.feature_extractor.sampling_rate

        mel_filters = torch.from_numpy(self.feature_extractor.mel_filters).float()
        self.register_buffer("mel_filters", mel_filters)

        hann_window = torch.hann_window(self.n_fft)
        self.register_buffer("hann_window", hann_window)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.hann_window,  # type: ignore
            return_complex=True,
        )
        magnitude = torch.abs(stft)
        mel_spec = self.mel_filters.T @ magnitude
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-10))
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        # Static padding to fixed size [80, 3000] — CoreML doesn't support dynamic 2D padding
        # We pad with a fixed conservative amount and then slice to exactly 3000
        log_mel = torch.nn.functional.pad(
            log_mel, (0, 3000), mode="constant", value=0
        )
        log_mel = log_mel[:, :3000]

        return log_mel


class EncoderWrapper(torch.nn.Module):
    def __init__(
        self, preprocessor: ProcessorTorchModule, encoder: torch.nn.Module
    ) -> None:
        super().__init__()
        self._preprocessor = preprocessor
        self._encoder = encoder
        self._n_fft = 512
        self._encoder_et_program: Optional[Program] = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        log_mel = self._preprocessor(waveform)  # type: ignore
        assert log_mel.shape == torch.Size([80, 3000]), (
            "Input features have to be of size [80, 3000]!"
        )
        return self._encoder(log_mel.unsqueeze(0)).get("last_hidden_state")


class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module, proj_out: torch.nn.Module):
        super().__init__()
        self.decoder = decoder
        self.proj_out = proj_out

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.decoder.forward(
            input_ids=input_ids,
            encoder_hidden_states=encoder_output,
            use_cache=False,
        )
        logits = self.proj_out(output[0])
        return logits


class ExecuTorchModule:
    def __init__(self, pte_path: str) -> None:
        self._runtime = Runtime.get()
        self._program = self._runtime.load_program(Path(pte_path))
        self._forward_method = self._program.load_method("forward")

    def __call__(self, args: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        return self._forward_method.execute(args)  # type: ignore

    def forward(self, args: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        return self.__call__(args)


class WhisperExporter:
    def __init__(
        self,
        wrapped_encoder: EncoderWrapper,
        wrapped_decoder: DecoderWrapper,
    ) -> None:
        self.wrapped_encoder = wrapped_encoder
        self.wrapped_decoder = wrapped_decoder
        self.encoder_random_input = torch.rand(SR_INT * 25)  # 25 seconds of audio
        self.calibration_dataset = []  # Not loaded eagerly; only needed for generate_with_exported()
        self.encoder_et_module: Optional[ExecuTorchModule] = None
        self.decoder_et_module: Optional[ExecuTorchModule] = None

    def _get_encoder_calibration_dataset(self, num_samples: int) -> list[torch.Tensor]:
        hf_dataset = load_dataset(
            "librispeech_asr",
            "clean",
            split="validation",
            streaming=True,
        )
        audio_tensors = []
        target_sr = SR_INT
        for i, sample in enumerate(hf_dataset):
            if i > num_samples:
                break
            audio_dict = sample.get("audio", {})
            audio_array = audio_dict.get("array")
            audio_sr = audio_dict.get("sampling_rate")
            if audio_array is None or audio_sr is None:
                logger.error(
                    f"Failed to get audio / SR dict key for sample no. {i}, sample: {sample}"
                )

            audio_tensor = AudioUtils.load_audio_at_sr(
                audio_array, orig_sr=audio_sr, target_sr=target_sr
            ).float()
            audio_tensors.append(audio_tensor)
        return audio_tensors

    @torch.no_grad()
    def export_encoder(
        self,
        out_path: str,
        compute_unit: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_GPU,
    ) -> None:
        logger.info(f"Exporting encoder with CoreML backend (compute_unit={compute_unit})...")

        dynamic_shapes = {
            "waveform": {0: Dim("waveform_length", min=201, max=48 * SR_INT)}
        }
        example_args = (self.encoder_random_input,)

        torch_exported = torch.export.export(
            self.wrapped_encoder, args=example_args, dynamic_shapes=dynamic_shapes
        )

        coreml_partitioner = CoreMLPartitioner(
            compile_specs=CoreMLBackend.generate_compile_specs(
                compute_unit=compute_unit,
                minimum_deployment_target=ct.target.iOS17,
                # Use CoreML's native weight compression instead of torchao
                compute_precision=ct.precision.FLOAT16,
            )
        )

        et_program = to_edge_transform_and_lower(
            torch_exported,
            partitioner=[coreml_partitioner],
            compile_config=EdgeCompileConfig(
                _core_aten_ops_exception_list=[torch.ops.aten.unfold.default],
                _check_ir_validity=False,
            ),
        ).to_executorch()

        logger.info(f"Successfully exported encoder, saving to {out_path}...")
        with open(out_path, "wb") as f:
            et_program.write_to_file(f)

        self.encoder_et_module = ExecuTorchModule(out_path)

    @torch.no_grad()
    def export_decoder(
        self,
        out_path: str,
        compute_unit: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_GPU,
    ) -> None:
        if not self.encoder_et_module:
            raise RuntimeError(
                "Can't run decoder export without running encoder export prior!"
            )
        sample_encoder_output = self.encoder_et_module((self.encoder_random_input,))[0]

        # CoreML does not support dynamic shapes (SymInt) — export with static shape
        # We export multiple decoders for different sequence lengths and let the
        # runtime pick the right one, or just use MAX_SEQ_LEN as a fixed size
        example_sequence = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.int64)

        example_args = (
            example_sequence,
            sample_encoder_output,
        )

        torch_exported = torch.export.export(
            self.wrapped_decoder, args=example_args
        )

        coreml_partitioner = CoreMLPartitioner(
            compile_specs=CoreMLBackend.generate_compile_specs(
                compute_unit=compute_unit,
                minimum_deployment_target=ct.target.iOS17,
                compute_precision=ct.precision.FLOAT16,
            )
        )

        et_program = to_edge_transform_and_lower(
            torch_exported,
            partitioner=[coreml_partitioner],
        ).to_executorch()

        logger.info(f"Successfully exported decoder, saving to {out_path}...")
        with open(out_path, "wb") as f:
            et_program.write_to_file(f)

        self.decoder_et_module = ExecuTorchModule(out_path)

    def test_exported_encoder(self) -> None:
        if not self.encoder_et_module:
            raise RuntimeError(
                "Can't test the exported encoder, ETModule is not loaded!"
            )
        wrapped_encoder_output = self.wrapped_encoder(self.encoder_random_input)
        et_encoder_output = self.encoder_et_module((self.encoder_random_input,))[0]
        hf_encoder_output = self.wrapped_encoder(self.encoder_random_input)
        assert torch.allclose(et_encoder_output, wrapped_encoder_output, atol=1e-8), (
            f"HF and ET encoder results do not match!, diff: {(et_encoder_output - hf_encoder_output)}"
        )

    def generate_with_exported(self, decode_fn: Callable) -> None:
        if not self.decoder_et_module or not self.encoder_et_module:
            raise RuntimeError("Decoder model must be exported before generation!")

        start_time = time.perf_counter()
        audio_sample = self.calibration_dataset[2]
        encoder_output = self.encoder_et_module((audio_sample,))[0]  # type: ignore
        encoding_time = (time.perf_counter() - start_time) * 1000

        logger.info(f"Encoder processing time: {encoding_time:.2f} ms")

        bos_token, eos_token = 50257, 50256
        input_ids = torch.tensor([[bos_token]], dtype=torch.int64)
        total_decoder_time = 0

        logger.info("Starting text generation...")
        for step in range(MAX_SEQ_LEN):
            start_time = time.perf_counter()
            result = self.decoder_et_module((input_ids, encoder_output))
            step_time = (time.perf_counter() - start_time) * 1000
            total_decoder_time += step_time

            next_token = torch.argmax(result[0][:, -1, :], dim=-1).item()
            if step % 10 == 0:
                logger.info(
                    f"Step {step}, token: {next_token}, time: {step_time:.2f} ms"
                )

            if next_token == eos_token:
                logger.info(f"EOS token reached at step {step}")
                break

            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

        generated_text = decode_fn(input_ids)
        steps_taken = step + 1  # type: ignore
        avg_decoder_time = total_decoder_time / steps_taken
        total_time = encoding_time + total_decoder_time

        logger.info(f"Generated text: {generated_text}")
        logger.info("=== Timing Summary ===")
        logger.info(f"Encoder: {encoding_time:.2f} ms")
        logger.info(
            f"Decoder: {total_decoder_time:.2f} ms ({steps_taken} steps, avg: {avg_decoder_time:.2f} ms/step)"
        )
        logger.info(
            f"Total: {total_time:.2f} ms, for audio length of: {(audio_sample.shape[0] / SR_INT):.2f}s"
        )


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


if __name__ == "__main__":
    model_name = "openai/whisper-tiny.en"
    hf_preprocessor = AutoProcessor.from_pretrained(model_name)
    hf_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en",
        generation_config=GENERATION_CONFIG,
    ).eval()

    wrapped_preprocessor = ProcessorTorchModule(hf_preprocessor)
    wrapped_encoder = EncoderWrapper(wrapped_preprocessor, hf_model.get_encoder())
    decoder = hf_model.get_decoder()

    decoder.embed_positions = WhisperPositionalEmbedding(decoder.embed_positions.weight)  # type: ignore

    # Skip embedding quantization — incompatible with CoreML runtime
    # embedding_transforms = get_quant_embedding_transform(
    #     "8,64", use_shared_embedding=True
    # )
    # decoder = embedding_transforms(decoder)
    wrapped_decoder = DecoderWrapper(decoder, hf_model.proj_out)

    # torchao quantization is incompatible with CoreML (int8 dtype not supported as input)
    # CoreML handles quantization internally via compute_precision in compile specs
    # wrapped_encoder = torchao_quantize(wrapped_encoder, group_size=64)
    # wrapped_decoder = torchao_quantize(wrapped_decoder, group_size=64)

    exporter = WhisperExporter(wrapped_encoder, wrapped_decoder)

    # Export CPU_AND_GPU variants (encoder must come first to set encoder_et_module)
    exporter.export_encoder(out_path=ENCODER_OUT_FILENAME_GPU, compute_unit=ct.ComputeUnit.CPU_AND_GPU)
    exporter.export_decoder(out_path=DECODER_OUT_FILENAME_GPU, compute_unit=ct.ComputeUnit.CPU_AND_GPU)

    # Export ALL variants (includes ANE/NPU)
    exporter.export_encoder(out_path=ENCODER_OUT_FILENAME_ALL, compute_unit=ct.ComputeUnit.ALL)
    exporter.export_decoder(out_path=DECODER_OUT_FILENAME_ALL, compute_unit=ct.ComputeUnit.ALL)