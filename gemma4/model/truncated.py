"""
truncated.py
============

Load the real Gemma 4 E2B weights into a **truncated** Gemma4EdgeForCausalLM
that keeps only the first N decoder layers (default 5).

The first 5 layers of the real config are
    [sliding, sliding, sliding, sliding, full]
so this truncation still exercises both attention variants and the
sliding/full RoPE/attention-mask code paths. KV sharing is off
(num_kv_shared_layers=0 — real model's 20 shared layers live beyond
index 14, so they don't exist in a 5-layer slice).

Usage:
    from gemma4.model.truncated import load_truncated_from_safetensors
    m = load_truncated_from_safetensors(
        "/path/to/model.safetensors", num_layers=5, dtype=torch.float16
    )

Because the safetensors file maps almost 1:1 onto the original config,
truncation is done with two slicing steps in the loaded state_dict:

  - `embed_tokens_per_layer.weight`  [V, L*d] → [V, N*d]   (drop col suffix)
  - `per_layer_model_projection.weight`  [L*d, H] → [N*d, H]   (drop row suffix)
  - drop any `model.layers.i.*` key where i >= N
"""
from __future__ import annotations
import os
from typing import Optional

import torch

from .text import (
    Gemma4EdgeConfig,
    Gemma4EdgeForCausalLM,
    _config_from_json,
)


def _truncated_config(full: Gemma4EdgeConfig, num_layers: int) -> Gemma4EdgeConfig:
    """Build a config that is identical to `full` but shorter.

    Preserves the double-wide MLP boundary from the full config so that
    truncations overlapping layer indices >= full.first_kv_shared get the
    12288 FFN width (matching the real checkpoint). We preserve this via
    the `num_kv_shared_layers` counter even though the truncated model
    itself does no KV sharing (source layers might be past the slice).
    """
    if num_layers < 1 or num_layers > full.num_hidden_layers:
        raise ValueError(f"num_layers must be in [1, {full.num_hidden_layers}]")

    # Keep the first N layer types; __post_init__ will force the last one
    # to "full_attention" so attention stays sensible.
    layer_types = list(full.layer_types[:num_layers])

    # In the real model, layers with idx >= first_kv_shared are
    # KV-shared AND use the double-wide (2x) MLP. The checkpoint's
    # gate_proj/up_proj/down_proj for those layers are sized at
    # intermediate_size * 2. We must preserve that boundary on the
    # truncated config so Gemma4EdgeConfig.ffn_intermediate_size() picks
    # the correct FFN width for each kept layer.
    full_first_kv_shared = full.num_hidden_layers - full.num_kv_shared_layers
    kv_shared_in_trunc = max(0, num_layers - full_first_kv_shared)

    return Gemma4EdgeConfig(
        vocab_size=full.vocab_size,
        hidden_size=full.hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=full.num_attention_heads,
        num_key_value_heads=full.num_key_value_heads,
        head_dim=full.head_dim,
        global_head_dim=full.global_head_dim,
        intermediate_size=full.intermediate_size,
        ple_dim=full.ple_dim,
        ple_vocab_size=full.ple_vocab_size,
        use_ple=full.use_ple,
        sliding_window=full.sliding_window,
        max_position_embeddings=full.max_position_embeddings,
        layer_types=layer_types,
        rope_theta=full.rope_theta,
        rope_local_base_freq=full.rope_local_base_freq,
        partial_rotary_factor=full.partial_rotary_factor,
        # Preserve double-wide MLP boundary: in the real 35-layer config
        # with num_kv_shared_layers=20, layers 15..34 are double-wide.
        # A truncation of N layers includes max(0, N-15) of those, so we
        # keep the same count here (so first_kv_shared = N - this = 15
        # when N >= 15, matching the real boundary).
        num_kv_shared_layers=kv_shared_in_trunc,
        use_double_wide_mlp=full.use_double_wide_mlp,
        final_logit_softcapping=full.final_logit_softcapping,
        tie_word_embeddings=full.tie_word_embeddings,
    )


def load_truncated_from_safetensors(
    safetensors_path: str,
    num_layers: int = 5,
    config: Optional[Gemma4EdgeConfig] = None,
    dtype: torch.dtype = torch.float16,
) -> Gemma4EdgeForCausalLM:
    """Load the real-weights checkpoint into an N-layer truncated model.

    Unlike `load_from_safetensors`, this skips `model.layers.i.*` tensors for
    i >= num_layers (big speed/memory win) and slices the PLE tensors so the
    truncated model's parameter shapes match.
    """
    from safetensors import safe_open

    # --- 1. Full config (just to pull dims), then truncate ---
    if config is None:
        search_dirs = [
            os.path.dirname(os.path.abspath(safetensors_path)),
            os.path.dirname(os.path.dirname(os.path.abspath(safetensors_path))),
            os.getcwd(),
        ]
        config_json = None
        for d in search_dirs:
            candidate = os.path.join(d, "config.json")
            if os.path.exists(candidate):
                config_json = candidate
                break
        full_cfg = _config_from_json(config_json) if config_json else Gemma4EdgeConfig()
    else:
        full_cfg = config

    trunc_cfg = _truncated_config(full_cfg, num_layers)
    print(f"  Truncating to {num_layers} layers "
          f"(layer_types={trunc_cfg.layer_types})")

    model = Gemma4EdgeForCausalLM(trunc_cfg)

    d = full_cfg.ple_dim
    N = num_layers
    kept_layer_prefix = tuple(f"layers.{i}." for i in range(N))

    state_dict: dict = {}
    prefix = "model.language_model."
    with safe_open(safetensors_path, framework="pt") as f:
        for key in f.keys():
            if not key.startswith(prefix):
                continue
            new_key = "model." + key[len(prefix):]

            # Drop layers beyond N
            if ".layers." in new_key:
                after = new_key.split(".layers.", 1)[1]  # e.g. "7.self_attn.q_proj.weight"
                layer_idx = int(after.split(".", 1)[0])
                if layer_idx >= N:
                    continue

            t = f.get_tensor(key)

            # PLE tensors: slice the layer-striped dim
            if new_key.endswith("embed_tokens_per_layer.weight"):
                # [V, L*d] → [V, N*d]
                t = t[:, : N * d].contiguous()
            elif new_key.endswith("per_layer_model_projection.weight"):
                # [L*d, H] → [N*d, H]
                t = t[: N * d, :].contiguous()

            state_dict[new_key] = t

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    expected_missing = set()
    if trunc_cfg.tie_word_embeddings:
        expected_missing.add("lm_head.weight")

    real_missing = [k for k in missing if k not in expected_missing]
    if real_missing:
        print(f"  WARNING: Missing keys: {real_missing}")
    if unexpected:
        print(f"  WARNING: Unexpected keys: {unexpected}")

    return model.to(dtype)
