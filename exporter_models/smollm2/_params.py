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

`quant` is the quantization recipe: `qmode` (default `8da4w`), linear
`group_size` and `embedding_quantize` "<bitwidth>,<groupsize>" (groupsize 0 =
per-channel). `None` leaves a knob at its executorch default.

**135m and 360m ship int8 per-channel (`8da8w`), not 4-bit.** SmolLM2 is too
small to survive 4-bit: at 135m the int4 build answers a "reverse a string"
prompt with nonsense `re.sub` code, and at 360m it refuses ordinary questions
outright ("I'm sorry, I'm not a six-year-old..."), while int8 tracks the bf16
answers on the same prompts. int8 is also still smaller than the bf16 it
replaces (0.166 vs 0.270 GB at 135m, 0.412 vs 0.725 GB at 360m), so the
quantized variant keeps its reason to exist. Measured on device -- see NOTES.md.

Note `group_size` must be 0 here: executorch's XNNPACK delegate supports group
quantization only at 4 bits ("Only 4b group quantization is supported"), so int8
has to be per-channel. `qmode` also has to be plain `8da8w` rather than
`torchao:8da8w`, which needs TorchAO's opaque kernel library and does not lower
to XNNPACK.

1_7b keeps 4-bit (`gs64 + emb 8,64`, the sweep's winner); it is large enough
that int4 holds up, and there `gs32` diverges outright (ppl 4e9).
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
        "quant": {"qmode": "8da8w", "group_size": 0, "embedding_quantize": "8,0"},
    },
    "360m": {
        "model_class": "smollm2",
        "source": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "params": {
            **_COMMON, "dim": 960, "hidden_dim": 2560, "n_heads": 15,
            "n_kv_heads": 5, "n_layers": 32, "rope_theta": 100000.0,
        },
        "quant": {"qmode": "8da8w", "group_size": 0, "embedding_quantize": "8,0"},
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
