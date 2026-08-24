"""LFM2.5 text per-size params + export-llm model class.

Params match the canonical executorch configs
(examples/models/lfm2/config/lfm2_5_{350m,1_2b}_config.json). The two sizes
share layer_types/vocab and differ only in dim / hidden_dim / n_heads.
"""

_LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv", "conv", "full_attention",
    "conv", "conv", "full_attention", "conv", "full_attention", "conv",
    "full_attention", "conv", "full_attention", "conv",
]

_COMMON = {
    "ffn_dim_multiplier": 1, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5,
    "rope_theta": 1000000.0, "use_scaled_rope": False, "vocab_size": 65536,
    "use_hf_rope": True, "use_qk_norm": True, "qk_norm_before_rope": True,
    "layer_types": _LAYER_TYPES,
}

# `quant` is the XNNPACK 8da4w recipe: linear group size + `embedding_quantize`
# "<bitwidth>,<groupsize>" (groupsize 0 = per-channel). These are the per-size
# winners of the quantization sweep the old `exports/llms` directory ran (size /
# throughput / perplexity / KL-divergence Pareto over fp16 and nine 8da4w
# group-size combinations); the first export of this model shipped a uniform
# group_size 64 with no embedding quantization instead, so the published 8da4w
# .pte predate this recipe. See NOTES.md.
SIZES = {
    "350m": {
        "model_class": "lfm2_5_350m",
        "params": {**_COMMON, "dim": 1024, "hidden_dim": 4608, "n_heads": 16},
        "quant": {"group_size": 32, "embedding_quantize": "8,32"},
    },
    "1_2b": {
        "model_class": "lfm2_5_1_2b",
        "params": {**_COMMON, "dim": 2048, "hidden_dim": 8192, "n_heads": 32},
        "quant": {"group_size": 32, "embedding_quantize": "8,0"},
    },
}

# The MLX backend quantizes to 4w on its own schedule -- unrelated to the
# XNNPACK 8da4w sweep above, so it keeps its own group size.
MLX_GROUP_SIZE = 64
