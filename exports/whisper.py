from time import perf_counter
import torch
import librosa
import torchaudio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperPreTrainedModel,
    GenerationConfig,
)
from torch.export import Dim
from typing import Callable, Optional

from executorch.exir import to_edge
from executorch.backends.xnnpack import XnnpackPartitioner


START_TOKEN = 50257
EOS_TOKEN = 50256
EXPORT_WITH_STATIC_CACHE = False
name = "whisper-tiny.en"
MODEL_NAME = f"openai/{name}"
EXAMPLE_AUDIO_PATH = "/Users/jakubchmura/Desktop/SWM_AI/executorch/audio_2.mp3"
MAX_SEQ_LEN = 128
GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    cache_implementation="static",
    max_length=MAX_SEQ_LEN,
    cache_config={
        "batch_size": 1,
        "max_cache_len": MAX_SEQ_LEN,
    },
)


class WhisperPreprocessor(torch.nn.Module):
    def __init__(self, mel_filters, n_fft=511):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("mel_filters", mel_filters)

    def pad(self, mel_spec):
        """Ensures output has shape (80, 3000) by padding or truncating along time axis (dim=1)."""
        target_freq, target_time = [80, 3000]
        current_freq, current_time = mel_spec.shape
        pad_amount = target_time - current_time
        mel_spec = torch.nn.functional.pad(
            mel_spec, (0, pad_amount), mode="constant", value=0
        )
        return mel_spec

    def normalize(self, mel_spec):
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_value_idx = torch.argmax(log_spec, dim=-1, keepdim=True)
        max_value = torch.gather(log_spec, dim=-1, index=max_value_idx)
        y = torch.where(log_spec < max_value - 8.0, max_value - 8.0, log_spec)
        y = torch.maximum(y, max_value - 8.0)
        y = (y + 4.0) / 4.0
        return y

    def forward(self, x):
        magnitudes = (10 ** (x / 20) * self.n_fft) ** 2
        magnitudes = magnitudes.transpose(1, 0)
        mel_filter_bank = self.mel_filters @ magnitudes
        return self.pad(self.normalize(mel_filter_bank))


class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder: WhisperPreTrainedModel, proj_out: torch.nn.Module):
        super().__init__()
        self.decoder = decoder
        self.proj_out = proj_out

    def forward(
        self, input_ids: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        output = self.decoder.forward(
            input_ids=input_ids, encoder_hidden_states=encoder_output, use_cache=False
        )
        logits = self.proj_out(output[0])
        return logits


class EncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        preprocessor: WhisperPreprocessor,
        encoder: torch.nn.Module,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(self, y):
        y = self.preprocessor.forward(y)
        return self.encoder.forward(y.unsqueeze(0))


def init_model(
    model_name: str, generation_config: Optional[GenerationConfig] = None
) -> WhisperPreTrainedModel:
    return WhisperForConditionalGeneration.from_pretrained(
        model_name, generation_config=generation_config
    )


def get_mel_from_filename(filename: str, processor: Callable) -> torch.Tensor:
    waveform, sr = torchaudio.load(filename, normalize=True)
    # Ensure the sample rate is 16kHz as expected by Whisper
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    # Process the waveform using the processor
    input_features = processor(
        waveform.squeeze(0), sampling_rate=sr, return_tensors="pt"
    ).input_features
    return input_features


def export_encoder(
    encoder: torch.nn.Module,
    to_xnnpack: Optional[bool] = True,
    output_filename: str = "whisper_encoder.pte",
) -> None:
    print("Exporting encoder...")
    dim = Dim("dim", min=1, max=3000)
    dynamic_shapes = {"y": {0: dim}}
    exported_program = torch.export.export(
        encoder, args=(torch.rand((1500, 256)),), dynamic_shapes=dynamic_shapes
    )
    edge_program = to_edge(exported_program)
    if to_xnnpack:
        edge_program = edge_program.to_backend(XnnpackPartitioner())
        output_filename = (
            name.replace("-", "_").replace(".", "_") + "_encoder_xnnpack.pte"
        )

    exec_program = edge_program.to_executorch()

    with open(output_filename, "wb") as file:
        exec_program.write_to_file(file)

    print("Encoder saved to file: {}".format(output_filename))


def export_decoder(
    decoder: torch.nn.Module,
    to_xnnpack: Optional[bool] = True,
    output_filename: str = "whisper_decoder.pte",
) -> None:
    print("Exporting decoder...")
    dim = Dim("dim1_x", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = {"input_ids": {1: dim}, "encoder_output": {}}
    exported_program = torch.export.export(
        decoder,
        args=(
            torch.randint(0, 10, [1, 10]).to(torch.int32),
            torch.rand((1, 1500, 384)),
        ),
        dynamic_shapes=dynamic_shapes,
    )
    edge_program = to_edge(exported_program)
    if to_xnnpack:
        edge_program = edge_program.to_backend(XnnpackPartitioner())
        output_filename = (
            name.replace("-", "_").replace(".", "_") + "_decoder_xnnpack.pte"
        )

    exec_program = edge_program.to_executorch()

    with open(output_filename, "wb") as file:
        exec_program.write_to_file(file)

    print("Decoder saved to file: {}".format(output_filename))


def generate(
    encoder: WhisperPreTrainedModel,
    decoder: WhisperPreTrainedModel,
    audio_filename: str,
    processor: Callable,
    proj_out: torch.nn.Module,
) -> torch.Tensor:
    def encoder_forward(
        model: WhisperPreTrainedModel,
        mel: torch.Tensor,
        return_last_hidden_state: bool = True,
    ) -> torch.Tensor:
        output = model.forward(input_features=mel)
        return output.last_hidden_state if return_last_hidden_state else output

    def decoder_forward(
        model: WhisperPreTrainedModel,
        prev_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        return_logits=False,
    ) -> torch.Tensor:
        output = model.forward(
            input_ids=prev_tokens, encoder_hidden_states=encoder_output
        )
        logits = proj_out(output[0])

        if return_logits:
            return logits
        next_token_ids = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return next_token_ids

    mel = get_mel_from_filename(audio_filename, processor)
    encoder_output = encoder_forward(encoder, mel)

    prev_tokens = torch.tensor([START_TOKEN], dtype=torch.int32).unsqueeze(0)

    for _ in range(MAX_SEQ_LEN):
        t1 = perf_counter()
        next_token_ids = decoder_forward(decoder, prev_tokens, encoder_output)
        print(perf_counter() - t1)
        prev_tokens = torch.cat([prev_tokens, next_token_ids], dim=1)
        if next_token_ids == EOS_TOKEN:
            break

    return prev_tokens


if __name__ == "__main__":
    model = (
        init_model(MODEL_NAME, GENERATION_CONFIG)
        if EXPORT_WITH_STATIC_CACHE
        else init_model(MODEL_NAME)
    )
    proj_out = model.proj_out
    n_fft = 511
    mel_filters_matrix = torch.from_numpy(
        librosa.filters.mel(sr=16e3, n_fft=n_fft, n_mels=80)
    )

    preprocessor = WhisperPreprocessor(mel_filters_matrix)

    # processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with torch.no_grad():
        export_encoder(EncoderWrapper(preprocessor, model.get_encoder()))
        export_decoder(DecoderWrapper(model.get_decoder(), proj_out))

    # with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION]), torch.no_grad():
    #     tokens = generate(encoder, decoder, EXAMPLE_AUDIO_PATH, processor, proj_out)

    # decoded_seq = processor.batch_decode(tokens, skip_special_tokens=True)
    # print(decoded_seq)
