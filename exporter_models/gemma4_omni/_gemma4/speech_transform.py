# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mel_filterbank(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    fft_length: int,
) -> torch.Tensor:
    all_freqs = torch.arange(n_freqs, dtype=torch.float32) * (sample_rate / fft_length)

    m_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
    m_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)

    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]

    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)

    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]

    fb = torch.maximum(torch.zeros(1), torch.minimum(down_slopes, up_slopes))
    return fb


class Gemma4SpeechTransformModel(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 320,
        hop_length: int = 160,
        fft_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        mel_floor: float = 0.001,
        input_scale_factor: float = 1.0,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fft_length = fft_length
        self.n_mels = n_mels
        self.mel_floor = mel_floor
        self.input_scale_factor = input_scale_factor

        window = torch.hann_window(frame_length, periodic=True)
        self.register_buffer("window", window)

        mel_filters = create_mel_filterbank(
            n_freqs=fft_length // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
            fft_length=fft_length,
        )
        self.register_buffer("mel_filters", mel_filters)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.input_scale_factor != 1.0:
            waveform = waveform * self.input_scale_factor

        pad_left = self.frame_length // 2
        waveform = F.pad(waveform, (pad_left, 0))

        frame_size = self.frame_length + 1
        frames = waveform.unfold(0, frame_size, self.hop_length)

        frames = frames[..., :-1]

        frames = frames * self.window

        stft = torch.fft.rfft(frames, n=self.fft_length, dim=-1)
        magnitude = torch.abs(stft)

        mel_spec = torch.matmul(magnitude, self.mel_filters)

        log_mel_spec = torch.log(mel_spec + self.mel_floor)

        return log_mel_spec
