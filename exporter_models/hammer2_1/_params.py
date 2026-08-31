"""Hammer2.1 per-size params, source checkpoints, and the shipped quant recipe.

Hammer2.1 is a function-calling fine-tune of Qwen2.5 and reuses that
architecture verbatim -- same executorch model classes, same HF->Meta weight
converter. Only the vocab differs (151665 vs 151936), so executorch ships no
configs for these; the params are carried here (they were vendored as
`jsons/hammer2_1/*.json` in the old `exports/llms` flow).

`model_class` only selects the eager model builder -- the architecture comes
from `params` and the weights from `checkpoint`, so 3B reuses the 1.5B class.

`quant` is the 8da4w recipe: linear group size + `embedding_quantize`
"<bitwidth>,<groupsize>" (groupsize 0 = per-channel). Values are the per-size
winners of the quantization sweep the old `exports/llms` directory ran
(size / throughput / perplexity / KL-divergence Pareto over bf16 and nine
8da4w group-size combinations).
"""

_COMMON = {
    "ffn_dim_multiplier": 1,
    "n_kv_heads": 2,
    "norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "use_scaled_rope": False,
    "vocab_size": 151665,
    "use_hf_rope": True,
    "attention_qkv_bias": True,
}

SIZES = {
    "0_5b": {
        "model_class": "qwen2_5_0_5b",
        "source": "MadeAgents/Hammer2.1-0.5b",
        "params": {**_COMMON, "dim": 896, "hidden_dim": 4864, "n_heads": 14, "n_layers": 24},
        "quant": {"group_size": 32, "embedding_quantize": "8,0"},
    },
    "1_5b": {
        "model_class": "qwen2_5_1_5b",
        "source": "MadeAgents/Hammer2.1-1.5b",
        "params": {**_COMMON, "dim": 1536, "hidden_dim": 8960, "n_heads": 12, "n_layers": 28},
        "quant": {"group_size": 32, "embedding_quantize": "8,32"},
    },
    "3b": {
        "model_class": "qwen2_5_1_5b",
        "source": "MadeAgents/Hammer2.1-3b",
        "params": {**_COMMON, "dim": 2048, "hidden_dim": 11008, "n_heads": 16, "n_layers": 36},
        "quant": {"group_size": 32, "embedding_quantize": "8,0"},
    },
}
