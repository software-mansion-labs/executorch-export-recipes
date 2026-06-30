#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import types

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenEmbeddingModule(torch.nn.Module):
    def __init__(self, self_decoder, config):
        super().__init__()
        self.embed_tokens = self_decoder.embed_tokens
        self.embed_scale = self_decoder.embed_scale
        self.embed_tokens_per_layer = self_decoder.embed_tokens_per_layer
        self.embed_scale_per_layer = self_decoder.embed_scale_per_layer
        self.num_layers = config.num_hidden_layers
        self.ple_dim = config.hidden_size_per_layer_input
        self.vocab_per_layer = config.vocab_size_per_layer_input

    def forward(self, input_ids: torch.Tensor):
        embeds = self.embed_tokens(input_ids) * self.embed_scale

        mask = torch.logical_and(input_ids >= 0, input_ids < self.vocab_per_layer)
        toks = torch.where(mask, input_ids, torch.zeros_like(input_ids))
        ple = self.embed_tokens_per_layer(toks) * self.embed_scale_per_layer
        b, s = input_ids.shape
        ple = ple.view(b, s, self.num_layers, self.ple_dim)
        return embeds, ple


class TextDecoderModule(torch.nn.Module):
    def __init__(self, causal_lm):
        super().__init__()
        self.model = causal_lm.model
        sd = self.model.self_decoder
        self._proj = sd.per_layer_model_projection
        self._proj_scale = sd.per_layer_projection_scale
        self._proj_norm = sd.per_layer_projection_norm
        self._input_scale = sd.per_layer_input_scale
        self._num_layers = sd.config.num_hidden_layers
        self._ple_dim = sd.config.hidden_size_per_layer_input

    def _project_ple(self, inputs_embeds, ple_tok):
        proj = self._proj(inputs_embeds) * self._proj_scale
        b, s, _ = inputs_embeds.shape
        proj = proj.view(b, s, self._num_layers, self._ple_dim)
        proj = self._proj_norm(proj)
        combined = (proj + ple_tok) * self._input_scale
        return combined.permute(2, 0, 1, 3)

    def forward(self, inputs_embeds, ple_tok, input_pos):
        sd = self.model.self_decoder
        per_layer_inputs = self._project_ple(inputs_embeds, ple_tok)

        hidden_states = inputs_embeds
        shared_kv = {}
        for i, layer in enumerate(sd.layers):
            hidden_states, _, kv_to_share = layer(
                hidden_states=hidden_states,
                per_layer_input=per_layer_inputs[i],
                input_pos=input_pos,
            )
            if kv_to_share is not None:
                shared_kv[i] = kv_to_share

        remaining = per_layer_inputs[sd.num_layers :]
        hidden_states = self.model.cross_decoder(
            hidden_states=hidden_states,
            per_layer_inputs=remaining,
            shared_kv=shared_kv,
            input_pos=input_pos,
        )
        hidden_states = self.model.norm(hidden_states)
        hidden_states = hidden_states[:, -1:, :]
        logits = self.model.lm_head(hidden_states)
        if self.model.final_logit_softcapping > 0:
            from .text_decoder.gemma4_transformer import (
                scaled_tanh,
            )

            logits = scaled_tanh(logits, self.model.final_logit_softcapping)
        return logits


def _build_text_model(checkpoint_path: str, max_seq_len: int, group_size: int = 32):
    from executorch.backends.mlx.llm.quantization import quantize_model_
    from .text_decoder.gemma4_config import (
        Gemma4Config,
    )
    from .text_decoder.gemma4_model import Gemma4Model
    from .text_decoder.mlx_source_transformations import (
        mlx_source_transformations,
    )

    config = Gemma4Config.from_config("e2b")
    config.use_kv_cache = True
    config.max_seq_len = max_seq_len
    config.enable_dynamic_shape = True
    config.use_custom_sdpa = False

    mw = Gemma4Model(config=config, checkpoint_path=checkpoint_path, dtype=torch.float32)
    model = mw.get_eager_model()
    model.eval()
    mlx_source_transformations(model, dtype=torch.float32)
    quantize_model_(
        model, qlinear_config="4w", qlinear_group_size=group_size,
        qembedding_config="4w", qembedding_group_size=group_size,
    )
    model.eval()
    return model, config
