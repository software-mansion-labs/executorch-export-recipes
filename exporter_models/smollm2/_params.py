"""SmolLM2 per-size params, source checkpoints, and the shipped quant recipe.

135M params match the canonical executorch config
(examples/models/smollm2/135M_config.json); executorch ships no 360M or 1.7B
config, so those params are carried here (they were vendored as
`jsons/smollm2/*.json` in the old `exports/llms` flow).

All three sizes share the single `smollm2` model class -- the architecture comes
from `params` and the weights from `checkpoint`. Note that 1.7B is not a scaled
135M: it drops GQA (n_kv_heads == n_heads) and uses a different rope theta.

`source` is the upstream HuggingFace checkpoint the export converts to Meta
format. These are the *Instruct* repos; export_llama's own auto-download would
resolve `smollm2` to the 135M base repo for every size, which is why the
checkpoint is always passed explicitly.

`quant` is the 8da4w recipe: linear group size + `embedding_quantize`
"<bitwidth>,<groupsize>" (groupsize 0 = per-channel); `None` means leave the
knob at its executorch default. Values are the per-size winners of the
quantization sweep the old `exports/llms` directory ran (size / throughput /
perplexity / KL-divergence Pareto over bf16 and nine 8da4w group-size
combinations), with one deliberate correction:

**135m and 360m do NOT use the sweep's `8da4w_default` winner.** An unset
`group_size` makes executorch fall back to 128, and it then quantizes a Linear
only when `in_features % group_size == 0`. At dim 576 / 960 that skips every
attention projection and two thirds of the MLP, so "8da4w_default" is very
nearly the unquantized model -- which is why it trivially won the sweep's
quality metrics, and why the currently-published `_8da4w.pte` for these two
sizes are *larger* than their bf16 counterparts (0.56 vs 0.27 GB, 1.36 vs
0.72 GB). 576 and 960 are both divisible by 64, so `group_size 64` quantizes
every linear. See NOTES.md.
"""

_COMMON = {
    "ffn_dim_multiplier": 1,
    "norm_eps": 1e-05,
    "use_scaled_rope": False,
    "vocab_size": 49152,
    "use_hf_rope": False,
    "attention_qkv_bias": False,
}

SIZES = {
    "135m": {
        "model_class": "smollm2",
        "source": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "params": {
            **_COMMON, "dim": 576, "hidden_dim": 1536, "n_heads": 9,
            "n_kv_heads": 3, "n_layers": 30, "rope_theta": 100000.0,
        },
        "quant": {"group_size": 64, "embedding_quantize": None},
    },
    "360m": {
        "model_class": "smollm2",
        "source": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "params": {
            **_COMMON, "dim": 960, "hidden_dim": 2560, "n_heads": 15,
            "n_kv_heads": 5, "n_layers": 32, "rope_theta": 100000.0,
        },
        "quant": {"group_size": 64, "embedding_quantize": None},
    },
    "1_7b": {
        "model_class": "smollm2",
        "source": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params": {
            **_COMMON, "dim": 2048, "hidden_dim": 8192, "n_heads": 32,
            "n_kv_heads": 32, "n_layers": 24, "rope_theta": 130000.0,
        },
        "quant": {"group_size": 64, "embedding_quantize": "8,64"},
    },
}
