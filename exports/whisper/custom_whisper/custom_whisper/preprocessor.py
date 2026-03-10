from transformers import AutoProcessor
import numpy as np
import torch
import torch.nn as nn


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

        # Precompute STFT kernels for XNNPACK compatibility (avoids complex tensors and non-delegated STFT)
        k = torch.arange(self.n_fft // 2 + 1).view(-1, 1)
        n = torch.arange(self.n_fft).view(1, -1)
        cos_kernel = torch.cos(2 * np.pi * k * n / self.n_fft) * self.hann_window
        sin_kernel = -torch.sin(2 * np.pi * k * n / self.n_fft) * self.hann_window
        # Kernel shape: [out_channels, 1, kernel_size] = [2 * (n_fft//2 + 1), 1, n_fft]
        stft_kernel = torch.stack([cos_kernel, sin_kernel], dim=1).reshape(-1, 1, self.n_fft)
        self.register_buffer("stft_kernel", stft_kernel)

    def pad(self, mel_spec):
        """Ensures output has shape (80, 3000) by padding or truncating along time axis (dim=1)."""
        _, target_time = [80, 3000]
        _, current_time = mel_spec.shape
        pad_amount = target_time - current_time + 1 # Add 1 to fix the issue of empty padding (Out tensor is too small for the copy operation.)
        mel_spec = torch.nn.functional.pad(
            mel_spec, (0, pad_amount), mode="constant", value=0
        )
        return mel_spec[:, :target_time]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # 1. Compute STFT using Conv1d for XNNPACK delegation (avoids non-delegated complex ops)
        padding = self.n_fft // 2
        # Use constant padding for widest compatibility; reflect padding is often not delegated
        x = torch.nn.functional.pad(waveform.view(1, 1, -1), (padding, padding), mode="constant", value=0)
        
        # Conv1d gives us real and imaginary parts as separate channels
        # Output shape: [1, 2 * (n_fft//2 + 1), time_frames]
        stft = torch.nn.functional.conv1d(x, self.stft_kernel, stride=self.hop_length)
        
        # 2. Get magnitude spectrogram: sqrt(Real^2 + Imag^2)
        # Reshape to [n_fft//2 + 1, 2, time_frames]
        stft = stft.view(self.n_fft // 2 + 1, 2, -1)
        magnitude = torch.sqrt(stft.pow(2).sum(dim=1) + 1e-10)

        # 3. Apply mel filter bank
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
