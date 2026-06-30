#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATCH_SIZE = 16
VIS_H = 448
VIS_W = 640
VIS_NY = VIS_H // PATCH_SIZE  # 28
VIS_NX = VIS_W // PATCH_SIZE  # 40
NUM_PATCHES = VIS_NY * VIS_NX  # 1120
SOFT_TOKENS = 280  # pooled 2x2 by positions

BLOCK_SAMPLES = 7680
AUDIO_K_MIN = 1
AUDIO_K_MAX = 62
AUDIO_K_TRACE = 2
MEL_HOP = 160


class RunnerVisionEncoder(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        gx = torch.arange(VIS_NX).repeat(VIS_NY)
        gy = torch.arange(VIS_NY).repeat_interleave(VIS_NX)
        self.register_buffer(
            "pos_ids",
            torch.stack([gx, gy], dim=-1).unsqueeze(0).long(),
            persistent=False,
        )
        self.register_buffer(
            "padding", torch.zeros(1, NUM_PATCHES, dtype=torch.bool), persistent=False
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image * (1.0 / 255.0)
        p = PATCH_SIZE
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 4, 5, 1).contiguous()
        patches = x.flatten(3).flatten(1, 2)  # [1, 1120, 768]

        inner = self.inner
        inputs_embeds = inner.patch_embedder(patches, self.pos_ids, self.padding)
        enc_out = inner.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~self.padding,
            pixel_position_ids=self.pos_ids,
        )
        hidden, pooler_mask = inner.pooler(
            hidden_states=enc_out.last_hidden_state,
            pixel_position_ids=self.pos_ids,
            padding_positions=self.padding,
            output_length=SOFT_TOKENS,
        )
        if inner.standardize:
            hidden = (hidden - inner.std_bias) * inner.std_scale
            hidden = hidden.masked_fill(~pooler_mask.unsqueeze(-1), 0.0)
        return inner.embed_vision(hidden)


class RunnerAudioEncoder(torch.nn.Module):
    def __init__(self, speech, audio_with_proj):
        super().__init__()
        self.speech = speech
        self.audio = audio_with_proj

    def forward(self, waveform: torch.Tensor, num_valid_samples: torch.Tensor):
        mel = self.speech(waveform[0])
        mel = mel.reshape(-1, mel.shape[-1])
        mel = torch.nn.functional.pad(mel, (0, 0, 0, 1))
        n_frames = mel.shape[0]
        valid_frames = (num_valid_samples // MEL_HOP - 1).clamp(min=1)
        frame_valid = torch.arange(n_frames) < valid_frames
        mel = mel * frame_valid.unsqueeze(-1).float()
        return self.audio(mel.unsqueeze(0), frame_valid.unsqueeze(0))


def _build_vision_runner_ep(checkpoint_path: str, group_size: int):
    from .export_gemma4 import (
        _load_hf_vision_encoder,
        _quantize_position_embedding_table,
    )
    from .export_gemma4_encoders_mlx import (
        _mlx_sdpa_attention,
        _quantize_int4,
    )
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import AttentionInterface

    inner = _load_hf_vision_encoder(checkpoint_path)
    if "mlx_sdpa" not in AttentionInterface._global_mapping:
        AttentionInterface.register("mlx_sdpa", _mlx_sdpa_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            "mlx_sdpa", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
        )
    inner.encoder.config._attn_implementation = "mlx_sdpa"
    _quantize_position_embedding_table(inner)
    _quantize_int4(inner, group_size)

    model = RunnerVisionEncoder(inner).eval()
    example = torch.zeros(1, 3, VIS_H, VIS_W)
    with torch.no_grad():
        ep = torch.export.export(model, (example,), strict=True)
    return model, ep


def _build_audio_runner_ep(checkpoint_path: str, group_size: int):
    from .export_gemma4 import (
        _HFAudioEncoderWithProjection,
        _load_hf_audio_encoder,
        _patch_hf_audio_for_export,
    )
    from .quant_utils import (
        apply_linear_quantization,
    )

    _patch_hf_audio_for_export()
    audio_with_proj = _load_hf_audio_encoder(checkpoint_path)
    assert isinstance(audio_with_proj, _HFAudioEncoderWithProjection)
    audio_with_proj = apply_linear_quantization(
        audio_with_proj, "8da4w", group_size=group_size
    )
    audio_with_proj.eval()

    import json
    import os

    from .speech_transform import (
        Gemma4SpeechTransformModel,
    )

    kwargs = {}
    processor_path = os.path.join(checkpoint_path, "processor_config.json")
    if os.path.exists(processor_path):
        with open(processor_path) as f:
            audio_fe = json.load(f).get("feature_extractor", {})
        field_map = {
            "sampling_rate": "sample_rate",
            "feature_size": "n_mels",
            "min_frequency": "f_min",
            "max_frequency": "f_max",
            "mel_floor": "mel_floor",
            "input_scale_factor": "input_scale_factor",
            "frame_length": "frame_length",
            "hop_length": "hop_length",
        }
        for hf_key, our_key in field_map.items():
            if hf_key in audio_fe:
                kwargs[our_key] = audio_fe[hf_key]
        logger.info(f"Speech transform config from checkpoint: {kwargs}")
    speech = Gemma4SpeechTransformModel(**kwargs).eval()
    assert speech.hop_length == MEL_HOP

    model = RunnerAudioEncoder(speech, audio_with_proj).eval()

    n_trace = BLOCK_SAMPLES * AUDIO_K_TRACE
    waveform = torch.zeros(1, n_trace)
    n_valid = torch.tensor(n_trace, dtype=torch.long)
    k_dim = torch.export.Dim("audio_k", min=AUDIO_K_MIN, max=AUDIO_K_MAX)
    with torch.no_grad():
        ep = torch.export.export(
            model,
            (waveform, n_valid),
            dynamic_shapes=({1: BLOCK_SAMPLES * k_dim}, None),
            strict=False,
        )
    return model, ep
