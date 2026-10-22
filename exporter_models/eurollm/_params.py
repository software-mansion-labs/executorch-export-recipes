"""EuroLLM per-size params, source checkpoint, and quant recipe.

EuroLLM is a plain Llama (GQA, SwiGLU, RMSNorm, RoPE theta 1e4) with an
UNTIED output head, so it exports through export_llama's LlamaTransformer
(`llama3` model class) with params carried here. Weights convert with the
qwen3 HF->Meta key map, which matches Llama's key names and keeps a separate
`lm_head`; SmolLM2's converter would tie `output` to the embedding.

`use_hf_rope` keeps HF's rotate-half RoPE, so the q/k projections need no
un-permuting.

`quant` is the published recipe: 8da8w with per-channel weights and an 8-bit
per-channel embedding table. 8da4w (gs32, the qwen3 1_7b recipe) loses 6-9 BLEU
on WMT13 en->de; see NOTES.md.

`vulkan_quant` is Vulkan's: int8 per-channel weight-only through the
VulkanQuantizer, which lowers to `_weight_int8pack_mm`. Vulkan fuses torchao
int8 weights into nothing, so `quant` would serialize a `dequantize_affine` the
runtime has no kernel for.
"""

SIZES = {
    "1_7b": {
        "model_class": "llama3",
        "source": "utter-project/EuroLLM-1.7B-Instruct",
        "params": {
            "dim": 2048,
            "hidden_dim": 5632,
            "n_heads": 16,
            "n_kv_heads": 8,
            "n_layers": 24,
            "vocab_size": 128000,
            "norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "use_scaled_rope": False,
            "use_hf_rope": True,
            "ffn_dim_multiplier": 1,
            "attention_qkv_bias": False,
        },
        "quant": {"qmode": "8da8w", "group_size": 0, "embedding_quantize": "8,0"},
        "vulkan_quant": {"pt2e_quantize": "vulkan_8w", "embedding_quantize": "8,0"},
    },
}
