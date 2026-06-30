#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _quantize_int4(model: torch.nn.Module, group_size: int = 32) -> None:
    from executorch.backends.mlx.llm.quantization import quantize_model_

    quantize_model_(
        model,
        qlinear_config="4w",
        qlinear_group_size=group_size,
        skip_incompatible_shapes=True,
    )
    model.eval()


def _lower_to_mlx(ep, output_path: str) -> None:
    import os

    from executorch import exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    method = "forward"
    edge = exir.to_edge_transform_and_lower(
        {method: ep},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    )
    et = edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
        )
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et.buffer)
    logger.info(f"Saved {output_path} ({len(et.buffer) / 1024 / 1024:.1f} MB)")


def _mlx_sdpa_attention(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs
):
    import executorch.backends.mlx.custom_ops  # noqa: F401

    out = torch.ops.mlx.custom_sdpa(
        query,
        key,
        value,
        0,
        attention_mask,
        0.0,
        False,
        scaling if scaling is not None else query.shape[-1] ** -0.5,
    )
    return out.transpose(1, 2).contiguous(), None


def _build_vision_ep(checkpoint_path: str, group_size: int):
    from .export_gemma4 import (
        _load_hf_vision_encoder,
        _quantize_position_embedding_table,
    )
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import AttentionInterface

    model = _load_hf_vision_encoder(checkpoint_path)
    if "mlx_sdpa" not in AttentionInterface._global_mapping:
        AttentionInterface.register("mlx_sdpa", _mlx_sdpa_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            "mlx_sdpa", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
        )
    model.encoder.config._attn_implementation = "mlx_sdpa"
    _quantize_position_embedding_table(model)
    _quantize_int4(model, group_size)

    max_patches = 2520
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
    return model, ep, (example_pixels, example_positions)


def _build_audio_ep(checkpoint_path: str, group_size: int):
    from .export_gemma4 import (
        _load_hf_audio_encoder,
        _patch_hf_audio_for_export,
    )

    _patch_hf_audio_for_export()
    model = _load_hf_audio_encoder(checkpoint_path)
    _quantize_int4(model, group_size)

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
    return model, ep, (example_features, example_mask)
