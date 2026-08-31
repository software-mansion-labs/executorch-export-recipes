"""Qwen2.5 per-size params, source checkpoints, and the shipped quant recipe.

0.5B and 1.5B params match the canonical executorch configs
(examples/models/qwen2_5/config/{0_5b,1_5b}_config.json); executorch ships no
3B config, so those params are carried here (they were vendored as
`jsons/qwen2_5/3b_config.json` in the old `exports/llms` flow).

`model_class` only selects the eager model builder -- the architecture comes
from `params` and the weights from `checkpoint`, so 3B reuses the 1.5B class.

`source` is the upstream HuggingFace checkpoint the export converts to Meta
format. These are the *Instruct* repos; export_llama's own auto-download would
resolve `qwen2_5_*` to the base (non-Instruct) repos instead, which is why the
checkpoint is always passed explicitly.

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
    "vocab_size": 151936,
    "use_hf_rope": True,
    "attention_qkv_bias": True,
}

SIZES = {
    "0_5b": {
        "model_class": "qwen2_5_0_5b",
        "source": "Qwen/Qwen2.5-0.5B-Instruct",
        "params": {**_COMMON, "dim": 896, "hidden_dim": 4864, "n_heads": 14, "n_layers": 24},
        "quant": {"group_size": 32, "embedding_quantize": "8,0"},
    },
    "1_5b": {
        "model_class": "qwen2_5_1_5b",
        "source": "Qwen/Qwen2.5-1.5B-Instruct",
        "params": {**_COMMON, "dim": 1536, "hidden_dim": 8960, "n_heads": 12, "n_layers": 28},
        "quant": {"group_size": 32, "embedding_quantize": "8,32"},
    },
    "3b": {
        "model_class": "qwen2_5_1_5b",
        "source": "Qwen/Qwen2.5-3B-Instruct",
        "params": {**_COMMON, "dim": 2048, "hidden_dim": 11008, "n_heads": 16, "n_layers": 36},
        "quant": {"group_size": 32, "embedding_quantize": "8,32"},
    },
}
