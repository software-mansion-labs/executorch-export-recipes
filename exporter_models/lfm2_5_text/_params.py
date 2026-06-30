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

SIZES = {
    "350m": {
        "model_class": "lfm2_5_350m",
        "params": {**_COMMON, "dim": 1024, "hidden_dim": 4608, "n_heads": 16},
    },
    "1_2b": {
        "model_class": "lfm2_5_1_2b",
        "params": {**_COMMON, "dim": 2048, "hidden_dim": 8192, "n_heads": 32},
    },
}

METADATA = '{"get_bos_id": 1, "get_eos_ids":[7]}'
GROUP_SIZE = 64
PREFILL_CAP = 64
CONTEXT_LEN = 2048
