"""Phi-4-mini params, source checkpoint, and the shipped quant recipe.

Params match the canonical executorch config
(examples/models/phi_4_mini/config/config.json). Phi-4-mini ships in one size
only, so this model has no `{size}` axis.

`source` is the upstream HuggingFace checkpoint the export converts to Meta
format. export_llama's own auto-download resolves `phi_4_mini` to the same repo,
but we pass the checkpoint explicitly so the source is pinned by this file
rather than by the executorch version.

`quant` is the 8da4w recipe: linear group size + `embedding_quantize`
"<bitwidth>,<groupsize>". Values are the winner of the quantization sweep the
old `exports/llms` directory ran (size / throughput / perplexity /
KL-divergence Pareto over bf16 and nine 8da4w group-size combinations).
"""

MODEL_CLASS = "phi_4_mini"
SOURCE = "microsoft/Phi-4-mini-instruct"
QUANT = {"group_size": 32, "embedding_quantize": "8,64"}

# The two rope-scaling tables are per-head-dim factors for Phi's partial rotary
# embedding; they are data, not tuning knobs.
_ROPE_SCALING_SHORT_FACTOR = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0
]

_ROPE_SCALING_LONG_FACTOR = [
    1.0, 1.118320672, 1.250641126, 1.398617824, 1.564103225, 1.74916897,
    1.956131817, 2.187582649, 2.446418898, 2.735880826, 3.059592084,
    3.421605075, 3.826451687, 4.279200023, 4.785517845, 5.351743533,
    5.984965424, 6.693110555, 7.485043894, 8.370679318, 9.36110372,
    10.4687158, 11.70738129, 13.09260651, 14.64173252, 16.37415215,
    18.31155283, 20.47818807, 22.90118105, 25.61086418, 28.64115884, 32.03,
    32.1, 32.13, 32.23, 32.6, 32.61, 32.64, 32.66, 32.7, 32.71, 32.93, 32.97,
    33.28, 33.49, 33.5, 44.16, 47.77
]

PARAMS = {
    'dim': 3072,
    'ffn_dim_multiplier': 1,
    'hidden_dim': 8192,
    'n_heads': 24,
    'n_kv_heads': 8,
    'n_layers': 32,
    'norm_eps': 1e-05,
    'rope_theta': 10000.0,
    'use_scaled_rope': False,
    'vocab_size': 200064,
    'use_hf_rope': True,
    'partial_rotary_factor': 0.75,
    'attention_qkv_bias': False,
    'original_max_position_embeddings': 4096,
    'max_position_embeddings': 131072,
    "rope_scaling_short_factor": _ROPE_SCALING_SHORT_FACTOR,
    "rope_scaling_long_factor": _ROPE_SCALING_LONG_FACTOR,
}
