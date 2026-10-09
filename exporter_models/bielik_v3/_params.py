"""Bielik v3 1.5B params, source checkpoint, and quant recipe."""

SOURCE = "speakleash/Bielik-1.5B-v3.0-Instruct"
MODEL_CLASS = "llama3_2"

PARAMS = {
    "dim": 1536,
    "hidden_dim": 8960,
    "n_heads": 12,
    "n_kv_heads": 2,
    "n_layers": 32,
    "norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "use_scaled_rope": False,
    "vocab_size": 32000,
    "use_hf_rope": True,
    "attention_qkv_bias": True,
    "ffn_dim_multiplier": 1,
}

QUANT = {"group_size": 32, "embedding_quantize": "8,0"}
