"""Torch-native port of Gemma4AudioFeatureExtractor._extract_spectrogram.

Replicates the exact algorithm used by HF's `Gemma4AudioFeatureExtractor`
so the PTE can own the log-mel frontend and accept raw fp32 waveform.

Algorithm (matches transformers/src/transformers/models/gemma4/
feature_extraction_gemma4.py as of 2026-04-24):

  * sampling_rate=16000, frame_length=320, hop_length=160
  * fft_length=512  (2 ** ceil(log2(320)))
  * periodic Hann window of length frame_length=320
  * HTK mel filter bank [257, 128], norm=None, 0..8kHz
  * semicausal padding: prepend frame_length//2=160 zeros; no right-pad
  * unfold with size=frame_length+1=321, step=hop_length=160
  * preemphasis=0 branch: drop last sample → frame shape 320
  * multiply window, rfft(n=512), abs, @ mel_filters, log(x + mel_floor=1e-3)
  * mask = attention_mask[frame_end_indices] where
      frame_end_indices = arange(T) * hop_length + frame_size_for_unfold - 1
    with frame_size_for_unfold = frame_length + 1 = 321
  * final output is log_mel * mask[..., None]  (zero out invalid frames)

The HTK filter bank is imported from `transformers.audio_utils.mel_filter_bank`
at module-build time and baked into a buffer. No torch op depends on the
Python-side computation at runtime.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.audio_utils import mel_filter_bank

SAMPLING_RATE = 16000
FRAME_LENGTH_MS = 20.0
HOP_LENGTH_MS = 10.0
FEATURE_SIZE = 128
MEL_FLOOR = 1e-3
MIN_FREQ = 0.0
MAX_FREQ = 8000.0

FRAME_LENGTH = int(round(SAMPLING_RATE * FRAME_LENGTH_MS / 1000.0))   # 320
HOP_LENGTH = int(round(SAMPLING_RATE * HOP_LENGTH_MS / 1000.0))        # 160
FFT_LENGTH = 2 ** math.ceil(math.log2(FRAME_LENGTH))                   # 512
NUM_FREQ_BINS = FFT_LENGTH // 2 + 1                                    # 257


def _periodic_hann(length: int) -> np.ndarray:
    # Matches transformers.audio_utils.window_function(periodic Hann).
    # w[n] = 0.5 - 0.5 * cos(2*pi*n / length)   for n in [0, length)
    n = np.arange(length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)).astype(np.float32)


class MelFrontend(nn.Module):
    """Stateless Gemma4 log-mel frontend: waveform[1, N] -> log_mel[1, T, 128].

    Also returns a boolean mask[1, T] indicating which mel frames correspond
    to entirely real-audio analysis windows (as opposed to padded zeros).
    The output log-mel is pre-multiplied by this mask, matching the HF
    `*mask[..., None]` step in `Gemma4AudioFeatureExtractor.__call__`.
    """

    def __init__(self):
        super().__init__()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mel_np = mel_filter_bank(
                num_frequency_bins=NUM_FREQ_BINS,
                num_mel_filters=FEATURE_SIZE,
                min_frequency=MIN_FREQ,
                max_frequency=MAX_FREQ,
                sampling_rate=SAMPLING_RATE,
                norm=None,
                mel_scale="htk",
            ).astype(np.float32)
        self.register_buffer(
            "mel_filters", torch.from_numpy(mel_np), persistent=True,
        )
        self.register_buffer(
            "window",
            torch.from_numpy(_periodic_hann(FRAME_LENGTH)),
            persistent=True,
        )
        self.frame_length = FRAME_LENGTH
        self.hop_length = HOP_LENGTH
        self.fft_length = FFT_LENGTH
        self.mel_floor = MEL_FLOOR

    def forward(self, waveform: torch.Tensor,
                num_valid_samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """waveform: [1, N_samples] fp32 mono 16 kHz. num_valid_samples: 0-d int64.

        Caller may zero-pad the waveform up to any length (to satisfy block
        alignment); num_valid_samples is the real PCM length. Frames whose
        analysis window extends past num_valid_samples are masked out.
        """
        pad_left = self.frame_length // 2            # 160
        frame_size_for_unfold = self.frame_length + 1  # 321

        wav = F.pad(waveform, (pad_left, 0))         # [1, N + 160]
        frames = wav.unfold(dimension=-1, size=frame_size_for_unfold, step=self.hop_length)
        frames = frames[..., :-1]
        frames = frames * self.window
        spec = torch.fft.rfft(frames, n=self.fft_length, dim=-1)
        mag = spec.abs()
        mel = torch.matmul(mag, self.mel_filters)
        log_mel = torch.log(mel + self.mel_floor)

        T = log_mel.shape[1]
        frame_end = (torch.arange(T, device=waveform.device) * self.hop_length
                     + frame_size_for_unfold - 1)
        # mask[i] = True iff frame i's analysis window stays within real audio.
        mask = frame_end < (pad_left + num_valid_samples.to(frame_end.dtype))
        mask = mask.unsqueeze(0)                     # [1, T]

        log_mel = log_mel * mask.unsqueeze(-1).to(log_mel.dtype)
        return log_mel, mask
