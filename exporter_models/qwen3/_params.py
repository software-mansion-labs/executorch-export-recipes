"""Qwen3 per-size params, source checkpoints, and the shipped quant recipe.

Params match the canonical executorch configs
(examples/models/qwen3/config/{0_6b,1_7b,4b}_config.json). The three sizes share
everything but dim / hidden_dim / n_heads / n_layers.

`source` is the upstream HuggingFace checkpoint the export converts to Meta
format. Qwen3 models are unified (no separate Instruct variant) -- the base
model supports both thinking and non-thinking modes. Note that export_llama's
own HUGGING_FACE_REPO_IDS would resolve the same ids here, but we pass the
checkpoint explicitly so the source is pinned by this file rather than by the
executorch version.

`quant` is the 8da4w recipe: linear group size + `embedding_quantize`
"<bitwidth>,<groupsize>" (groupsize 0 = per-channel). Values are the per-size
winners of the quantization sweep the old `exports/llms` directory ran
(size / throughput / perplexity / KL-divergence Pareto over bf16 and nine
8da4w group-size combinations).
"""

_COMMON = {
    "ffn_dim_multiplier": 1,
    "head_dim": 128,
    "n_kv_heads": 8,
    "norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "use_scaled_rope": False,
    "vocab_size": 151936,
    "use_hf_rope": True,
    "attention_qkv_bias": False,
    "use_qk_norm": True,
    "qk_norm_before_rope": True,
}

SIZES = {
    "0_6b": {
        "model_class": "qwen3_0_6b",
        "source": "Qwen/Qwen3-0.6B",
        "params": {**_COMMON, "dim": 1024, "hidden_dim": 3072, "n_heads": 16, "n_layers": 28},
        "quant": {"group_size": 32, "embedding_quantize": "8,64"},
    },
    "1_7b": {
        "model_class": "qwen3_1_7b",
        "source": "Qwen/Qwen3-1.7B",
        "params": {**_COMMON, "dim": 2048, "hidden_dim": 6144, "n_heads": 16, "n_layers": 28},
        "quant": {"group_size": 32, "embedding_quantize": "8,64"},
    },
    "4b": {
        "model_class": "qwen3_4b",
        "source": "Qwen/Qwen3-4B",
        "params": {**_COMMON, "dim": 2560, "hidden_dim": 9728, "n_heads": 32, "n_layers": 36},
        "quant": {"group_size": 32, "embedding_quantize": "8,64"},
    },
}
