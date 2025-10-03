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
import lovely_tensors.tensors as lt  # type: ignore

from executorch.backends.xnnpack import XnnpackPartitioner
import logging


logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)
lt.monkey_patch()

MODEL_NAME = "openai/whisper-tiny"
ENCODER_OUT_FILENAME = "whisper_encoder.pte"
DECODER_OUT_FILENAME = "whisper_decoder.pte"
SR_INT = int(16e3)
MAX_SEQ_LEN = 128
# FIXME: KV cache doesn't work. I strongly suggest spending some time on investigating this,
# as it believe it will be very beneficiial for the perf
# GENERATION_CONFIG = GenerationConfig(
#     use_cache=False,
#     cache_implementation="static",
#     max_length=MAX_SEQ_LEN,
#     cache_config={
#         "batch_size": 1,
#         "max_cache_len": MAX_SEQ_LEN,
#     },
# )
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
        # Extract preprocessing parameters
        self.n_fft = self.feature_extractor.n_fft
        self.hop_length = self.feature_extractor.hop_length
        self.n_mels = self.feature_extractor.feature_size
        self.sampling_rate = self.feature_extractor.sampling_rate

        # Create mel filter bank as a parameter
        # mel_filters should be [n_mels, n_fft//2 + 1] = [80, 201]
        mel_filters = torch.from_numpy(self.feature_extractor.mel_filters).float()
        self.register_buffer("mel_filters", mel_filters)

        # Register Hann window as buffer to avoid non-aten ops
        hann_window = torch.hann_window(self.n_fft)
        self.register_buffer("hann_window", hann_window)

    def pad(self, mel_spec):
        """Ensures output has shape (80, 3000) by padding or truncating along time axis (dim=1)."""
        _, target_time = [80, 3000]
        _, current_time = mel_spec.shape
        pad_amount = target_time - current_time
        mel_spec = torch.nn.functional.pad(
            mel_spec, (0, pad_amount), mode="constant", value=0
        )
        return mel_spec

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # 1. Compute STFT with pre-registered window
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.hann_window,  # type: ignore
            return_complex=True,
        )
        # 2. Get magnitude spectrogram
        # stft shape: [freq_bins, time_frames] = [201, time_frames]
        magnitude = torch.abs(stft)

        # 3. Apply mel filter bank
        # mel_filters: [80, 201], magnitude: [201, time_frames]
        # Result should be: [80, time_frames]
        mel_spec = self.mel_filters.T @ magnitude

        # 4. Convert to log scale
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-10))

        # 5. Normalize (Whisper-specific normalization)
        # reference for this: https://github.com/openai/whisper/blob/c0d2f624c09dc18e709e37c2ad90c039a4eb72a2/whisper/audio.py#L155
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        # 6. Pad to exactly 3000 time frames
        log_mel = self.pad(log_mel)

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
        self.calibration_dataset = self._get_encoder_calibration_dataset(100)
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
    def export_encoder(self, out_path="whisper_encoder_quantized.pte") -> None:
        logger.info("Exporting encoder...")

        dynamic_shapes = {
            "waveform": {0: Dim("waveform_length", min=201, max=48 * SR_INT)}
        }
        example_args = (self.encoder_random_input,)

        torch_exported = torch.export.export(
            self.wrapped_encoder, args=example_args, dynamic_shapes=dynamic_shapes
        )

        et_program = to_edge_transform_and_lower(
            torch_exported,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(
                _core_aten_ops_exception_list=[torch.ops.aten.unfold.default],
                _check_ir_validity=False,
            ),
        ).to_executorch()

        logger.info("Succesfully exported encoder, saving file...")
        with open(out_path, "wb") as f:
            et_program.write_to_file(f)

        self.encoder_et_module = ExecuTorchModule(out_path)

    @torch.no_grad()
    def export_decoder(
        self,
        out_path="whisper_decoder_quantized.pte",
    ):
        # Get the example output from decoder for sanity
        if not self.encoder_et_module:
            raise RuntimeError(
                "Can't run decoder export without running encoder export prior!"
            )
        sample_encoder_output = self.encoder_et_module((self.encoder_random_input,))[0]

        example_sequence = torch.tensor([[50256, 50257, 50258]], dtype=torch.int64)
        input_dim = Dim("input_seq_len", min=1, max=MAX_SEQ_LEN)

        dynamic_shapes = {
            "input_ids": {1: input_dim},
            "encoder_output": {},
        }

        example_args = (
            example_sequence,
            sample_encoder_output,
        )

        torch_exported = torch.export.export(
            self.wrapped_decoder, args=example_args, dynamic_shapes=dynamic_shapes
        )
        et_program = to_edge_transform_and_lower(
            torch_exported, partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        logger.info("Succesfully exported decoder, saving file...")
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

        # Generate text
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


if __name__ == "__main__":
    encoder_out_filename = "whisper_encoder.pte"
    decoder_out_filename = "whisper_decoder.pte"

    model_name = "openai/whisper-tiny.en"
    hf_preprocessor = AutoProcessor.from_pretrained(model_name)
    hf_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en",
        generation_config=GENERATION_CONFIG,
    ).eval()

    wrapped_preprocessor = ProcessorTorchModule(hf_preprocessor)
    wrapped_encoder = EncoderWrapper(wrapped_preprocessor, hf_model.get_encoder())
    decoder = hf_model.get_decoder()

    # WARNING: hacky stuff going on here, see comment above WhisperPositionalEmbedding
    decoder.embed_positions = WhisperPositionalEmbedding(decoder.embed_positions.weight)  # type: ignore

    # quantize embeddings
    embedding_transforms = get_quant_embedding_transform(
        "8,64", use_shared_embedding=True
    )
    decoder = embedding_transforms(decoder)
    wrapped_decoder = DecoderWrapper(decoder, hf_model.proj_out)

    wrapped_encoder = torchao_quantize(wrapped_encoder, group_size=64)
    wrapped_decoder = torchao_quantize(wrapped_decoder, group_size=64)

    exporter = WhisperExporter(wrapped_encoder, wrapped_decoder)

    exporter.export_encoder(out_path=encoder_out_filename)
    # exporter.test_exported_encoder()
    exporter.export_decoder(out_path=decoder_out_filename)
    # NOTE: Generation test currently disabled due to quantized embedding runtime issues
    # Use useSpeechToText to test the exported models
    # exporter.generate_with_exported(decode_fn=hf_preprocessor.batch_decode)
