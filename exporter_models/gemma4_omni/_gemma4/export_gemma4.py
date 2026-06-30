#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import functools
import gc
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _export_speech_transform(checkpoint_path: str):
    from .speech_transform import (
        Gemma4SpeechTransformModel,
    )

    kwargs = {}
    try:
        import json
        import os

        processor_path = os.path.join(checkpoint_path, "processor_config.json")
        if os.path.exists(processor_path):
            with open(processor_path, "r") as f:
                processor_config = json.load(f)
            audio_fe = processor_config.get("feature_extractor", {})
            if audio_fe:
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
    except Exception as e:
        logger.warning(
            f"Could not load audio config from checkpoint: {e}, using defaults"
        )

    model = Gemma4SpeechTransformModel(**kwargs)
    model.eval()

    sample_rate = kwargs.get("sample_rate", 16000)
    num_samples = sample_rate * 30
    example = torch.randn(num_samples)
    min_samples = model.frame_length + model.hop_length + 1
    waveform_dim = torch.export.Dim("waveform_length", min=min_samples, max=960000)

    with torch.no_grad():
        ep = torch.export.export(
            model, (example,), dynamic_shapes={"waveform": {0: waveform_dim}}
        )

    logger.info("Speech transform exported")
    del model
    gc.collect()
    return ep


class _HFAudioEncoderWithProjection(torch.nn.Module):
    def __init__(self, audio_tower, embed_audio):
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.audio_tower(features, attention_mask=mask)
        hidden = encoder_output.last_hidden_state
        output_mask = encoder_output.attention_mask
        return self.embed_audio(hidden), output_mask


def _patch_hf_audio_for_export():
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        raise ImportError(
            "transformers.models.gemma4 not found. Requires transformers >= 5.2.0."
        )

    cls = modeling_gemma4.Gemma4AudioRelPositionalEncoding
    if hasattr(cls.forward, "__wrapped__"):
        cls.forward = cls.forward.__wrapped__

    if hasattr(modeling_gemma4, "Gemma4AudioCausalConv1d"):
        conv_cls = modeling_gemma4.Gemma4AudioCausalConv1d
        cached = conv_cls.__dict__.get("left_pad")
        if isinstance(cached, functools.cached_property):
            conv_cls.left_pad = property(cached.func)


def _load_hf_conditional_model(checkpoint_path: str):
    import json
    import os
    import tempfile

    from transformers import AutoConfig

    config_path = os.path.join(checkpoint_path, "config.json")
    patched_path = checkpoint_path
    temp_dir = None

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        if "quantization_config" in config_dict:
            logger.info("  Patching config.json to remove quantization_config...")
            temp_dir = tempfile.mkdtemp(prefix="gemma4_export_")
            patched_path = temp_dir
            for item in os.listdir(checkpoint_path):
                src = os.path.join(checkpoint_path, item)
                dst = os.path.join(temp_dir, item)
                if item != "config.json":
                    os.symlink(src, dst)
            patched_config = {
                k: v for k, v in config_dict.items() if k != "quantization_config"
            }
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump(patched_config, f, indent=2)

    try:
        config = AutoConfig.from_pretrained(patched_path, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            config.quantization_config = None

        try:
            from transformers import Gemma4ForConditionalGeneration
        except ImportError:
            raise ImportError(
                "Gemma4ForConditionalGeneration not found. Requires transformers >= 5.2.0."
            )

        logger.info("  Loading HF Gemma4ForConditionalGeneration...")
        with torch.device("cpu"):
            model = Gemma4ForConditionalGeneration.from_pretrained(
                patched_path,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
        logger.info("  Model loaded")
        return model
    finally:
        if temp_dir is not None:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


def _load_hf_audio_encoder(checkpoint_path: str):
    model = _load_hf_conditional_model(checkpoint_path)

    audio_tower = model.model.audio_tower
    embed_audio = model.model.embed_audio
    logger.info(f"  Extracted audio_tower: {type(audio_tower).__name__}")
    logger.info(f"  Extracted embed_audio: {type(embed_audio).__name__}")

    wrapper = _HFAudioEncoderWithProjection(audio_tower, embed_audio)
    wrapper.eval()

    del model
    gc.collect()
    return wrapper


def _export_audio_encoder(checkpoint_path: str, quantize: str, group_size: int = 128):
    _patch_hf_audio_for_export()

    model = _load_hf_audio_encoder(checkpoint_path)

    if quantize != "none":
        from .quant_utils import (
            apply_linear_quantization,
        )

        model = apply_linear_quantization(model, quantize, group_size=group_size)
        model.eval()

    _num_frames = torch.export.Dim("_num_frames", min=2, max=63)
    num_frames_dim = 48 * _num_frames - 25
    valid_frames = 48 * 63 - 25
    example_features = torch.randn(1, valid_frames, 128)
    example_mask = torch.ones(1, valid_frames, dtype=torch.bool)

    with torch.no_grad():
        ep = torch.export.export(
            model,
            (example_features, example_mask),
            dynamic_shapes={
                "features": {1: num_frames_dim},
                "mask": {1: num_frames_dim},
            },
        )

    logger.info("Audio encoder exported")
    del model
    gc.collect()
    return ep


class _HFVisionEncoderWithProjection(torch.nn.Module):
    def __init__(self, vision_tower, embed_vision):
        super().__init__()
        self.patch_embedder = vision_tower.patch_embedder
        self.encoder = vision_tower.encoder
        self.pooler = vision_tower.pooler
        self.embed_vision = embed_vision
        self.standardize = getattr(vision_tower.config, "standardize", False)
        if self.standardize:
            self.std_bias = vision_tower.std_bias
            self.std_scale = vision_tower.std_scale
        self.pooling_kernel_size = vision_tower.config.pooling_kernel_size

    def forward(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pks = self.pooling_kernel_size
        output_length = pixel_values.shape[1] // (pks * pks)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)

        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )
        encoder_output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
        )

        hidden_states, pooler_mask = self.pooler(
            hidden_states=encoder_output.last_hidden_state,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale
            hidden_states = hidden_states.masked_fill(~pooler_mask.unsqueeze(-1), 0.0)

        embeddings = self.embed_vision(hidden_states)
        return embeddings, pooler_mask


def _load_hf_vision_encoder(checkpoint_path: str):
    model = _load_hf_conditional_model(checkpoint_path)

    vision_tower = model.model.vision_tower
    embed_vision = model.model.embed_vision
    logger.info(f"  Extracted vision_tower: {type(vision_tower).__name__}")
    logger.info(f"  Extracted embed_vision: {type(embed_vision).__name__}")

    wrapper = _HFVisionEncoderWithProjection(vision_tower, embed_vision)
    wrapper.eval()

    del model
    gc.collect()
    return wrapper


def _quantize_position_embedding_table(model):
    pet = model.patch_embedder.position_embedding_table
    original_mb = pet.numel() * 4 / (1024 * 1024)

    scale = pet.data.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    quantized = torch.round(pet.data / scale).clamp(-128, 127).to(torch.int8)

    del model.patch_embedder.position_embedding_table
    model.patch_embedder.register_buffer("_pet_int8", quantized)
    model.patch_embedder.register_buffer("_pet_scale", scale)

    new_mb = (quantized.numel() + scale.numel() * 4) / (1024 * 1024)
    logger.info(
        f"  Position embedding table: fp32 -> int8 per-channel ({original_mb:.0f} MB -> {new_mb:.0f} MB)"
    )

    import types

    def _position_embeddings_int8(self, pixel_position_ids, padding_positions):
        table = self._pet_int8.float() * self._pet_scale
        clamped = pixel_position_ids.clamp(min=0)
        one_hot = torch.nn.functional.one_hot(clamped, num_classes=table.shape[1])
        one_hot = one_hot.permute(0, 2, 1, 3).float()
        pos_emb = one_hot @ table
        pos_emb = pos_emb.sum(dim=1)
        return torch.where(padding_positions.unsqueeze(-1), 0.0, pos_emb)

    model.patch_embedder._position_embeddings = types.MethodType(
        _position_embeddings_int8, model.patch_embedder
    )


def _export_vision_encoder(
    checkpoint_path: str, quantize: str, group_size: int = 128, max_patches: int = 1260
):
    model = _load_hf_vision_encoder(checkpoint_path)

    _quantize_position_embedding_table(model)

    if quantize != "none":
        from .quant_utils import (
            apply_linear_quantization,
        )

        model = apply_linear_quantization(model, quantize, group_size=group_size)
        model.eval()

    _num_groups = torch.export.Dim("_num_groups", min=1, max=max_patches // 9)
    num_patches_dim = 9 * _num_groups

    example_pixels = torch.randn(1, max_patches, 3 * 16 * 16)
    example_positions = torch.zeros(1, max_patches, 2, dtype=torch.long)

    with torch.no_grad():
        ep = torch.export.export(
            model,
            (example_pixels, example_positions),
            dynamic_shapes={
                "pixel_values": {1: num_patches_dim},
                "pixel_position_ids": {1: num_patches_dim},
            },
        )

    logger.info("Vision encoder exported")
    del model
    gc.collect()
    return ep


def _export_text_decoder(
    checkpoint_path: str,
    quantize: str,
    max_seq_len: int,
    group_size: int = 128,
    tied_embedding: bool = False,
    variant: str = "e2b",
    quantize_kv_cache: bool = False,
    use_custom_sdpa: bool = True,
):
    from .quant_utils import (
        apply_embedding_quantization,
        apply_linear_quantization,
        parse_quantize,
    )
    from .text_decoder.gemma4_config import (
        Gemma4Config,
    )
    from .text_decoder.gemma4_model import Gemma4Model

    config = Gemma4Config.from_config(variant)
    config.use_kv_cache = True
    config.max_seq_len = max_seq_len
    config.enable_dynamic_shape = True
    config.use_custom_sdpa = use_custom_sdpa

    if use_custom_sdpa:
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

        logger.info("Custom SDPA enabled (tiled flash attention)")

    model_wrapper = Gemma4Model(
        config=config, checkpoint_path=checkpoint_path, dtype=torch.float32
    )
    model = model_wrapper.get_eager_model()
    model.eval()

    linear_quant, emb_quant = parse_quantize(quantize)
    if emb_quant and linear_quant and tied_embedding:
        from torchao.prototype.quantization.embedding.api import TiedEmbeddingQuantizer
        from torchao.quantization.granularity import PerAxis

        weight_dtype = torch.int4 if emb_quant == "emb4" else torch.int8
        logger.info(f"Applying tied embedding (embed_tokens + lm_head, {emb_quant})...")
        TiedEmbeddingQuantizer(
            weight_dtype=weight_dtype,
            granularity=PerAxis(0),
        ).quantize(
            model,
            embedding_to_unembedding={
                "model.self_decoder.embed_tokens": "model.lm_head",
            },
        )
        model = apply_embedding_quantization(model, emb_quant)
        model.eval()
    else:
        if emb_quant:
            model = apply_embedding_quantization(model, emb_quant)
            model.eval()
    if quantize_kv_cache:
        from .text_decoder.gemma4_attention import (
            replace_kv_cache_with_quantized_kv_cache,
        )

        logger.info("Replacing KV cache with INT8 quantized KV cache...")
        model = replace_kv_cache_with_quantized_kv_cache(
            model, use_custom_sdpa=use_custom_sdpa
        )
        model.eval()

    if linear_quant:
        model = apply_linear_quantization(model, linear_quant, group_size=group_size)
        model.eval()

    example_inputs = model_wrapper.get_example_inputs_with_audio(seq_len=770)
    dynamic_shapes = model_wrapper.get_dynamic_shapes(with_audio_embeds=True)

    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        with torch.no_grad():
            ep = torch.export.export(
                model,
                (example_inputs[0],),
                kwargs={
                    "input_pos": example_inputs[1],
                    "inputs_embeds": example_inputs[2],
                },
                dynamic_shapes=dynamic_shapes,
            )

    logger.info("Text decoder exported")
    del model, model_wrapper
    gc.collect()
    return ep
