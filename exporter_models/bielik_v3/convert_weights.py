from typing import Dict
import torch
from safetensors.torch import load_file


def convert_weights(checkpoint_dir_or_file: str, output_path: str) -> None:
    """Convert HuggingFace safetensors weights for Bielik-1.5B-v3.0 to Meta format."""
    import os
    if os.path.isdir(checkpoint_dir_or_file):
        safetensor_path = os.path.join(checkpoint_dir_or_file, "model.safetensors")
    else:
        safetensor_path = checkpoint_dir_or_file

    sd = load_file(safetensor_path)

    meta_sd = {}
    meta_sd["tok_embeddings.weight"] = sd["model.embed_tokens.weight"]
    meta_sd["norm.weight"] = sd["model.norm.weight"]
    meta_sd["output.weight"] = sd["lm_head.weight"]

    for i in range(32):
        prefix = f"model.layers.{i}."
        target = f"layers.{i}."
        meta_sd[f"{target}attention.wq.weight"] = sd[f"{prefix}self_attn.q_proj.weight"]
        meta_sd[f"{target}attention.wq.bias"] = sd[f"{prefix}self_attn.q_proj.bias"]
        meta_sd[f"{target}attention.wk.weight"] = sd[f"{prefix}self_attn.k_proj.weight"]
        meta_sd[f"{target}attention.wk.bias"] = sd[f"{prefix}self_attn.k_proj.bias"]
        meta_sd[f"{target}attention.wv.weight"] = sd[f"{prefix}self_attn.v_proj.weight"]
        meta_sd[f"{target}attention.wv.bias"] = sd[f"{prefix}self_attn.v_proj.bias"]
        meta_sd[f"{target}attention.wo.weight"] = sd[f"{prefix}self_attn.o_proj.weight"]
        if f"{prefix}self_attn.o_proj.bias" in sd:
            meta_sd[f"{target}attention.wo.bias"] = sd[f"{prefix}self_attn.o_proj.bias"]

        meta_sd[f"{target}attention_norm.weight"] = sd[f"{prefix}input_layernorm.weight"]
        meta_sd[f"{target}ffn_norm.weight"] = sd[f"{prefix}post_attention_layernorm.weight"]

        meta_sd[f"{target}feed_forward.w1.weight"] = sd[f"{prefix}mlp.gate_proj.weight"]
        meta_sd[f"{target}feed_forward.w2.weight"] = sd[f"{prefix}mlp.down_proj.weight"]
        meta_sd[f"{target}feed_forward.w3.weight"] = sd[f"{prefix}mlp.up_proj.weight"]

    torch.save(meta_sd, output_path)
