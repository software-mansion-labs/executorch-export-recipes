"""Hy-MT2 per-size params, source checkpoint, quant recipe, and weight converter.

Hy-MT2-1.8B is Tencent's HunYuan dense v1: a Llama layout (GQA, SwiGLU,
RMSNorm) plus a per-head RMSNorm on q and k applied AFTER RoPE, so it exports
through export_llama's LlamaTransformer (`llama3` model class) with
`use_qk_norm` and `qk_norm_before_rope=False`.

RoPE is HF's "dynamic" type with an NTK alpha but factor 1.0, which makes it
a fixed base: rope_theta * alpha ** (head_dim / (head_dim - 2)). That base is
baked into `rope_theta` here. `use_hf_rope` keeps HF's rotate-half RoPE, so
the q/k projections need no un-permuting.

`quant` holds the published XNNPACK recipes, keyed by variant precision:
8da4w (gs32, 4-bit gs32 embedding table; the default) and 8da8w (per-channel,
8-bit per-channel embedding table, as for EuroLLM). See NOTES.md.
"""

_HEAD_DIM = 128
_ROPE_ALPHA = 1000.0

SIZES = {
    "1_8b": {
        "model_class": "llama3",
        "source": "tencent/Hy-MT2-1.8B",
        "params": {
            "dim": 2048,
            "hidden_dim": 6144,
            "n_heads": 16,
            "n_kv_heads": 4,
            "head_dim": _HEAD_DIM,
            "n_layers": 32,
            "vocab_size": 120818,
            "norm_eps": 1e-05,
            "rope_theta": 10000.0 * _ROPE_ALPHA ** (_HEAD_DIM / (_HEAD_DIM - 2)),
            "use_scaled_rope": False,
            "use_hf_rope": True,
            "use_qk_norm": True,
            "qk_norm_before_rope": False,
            "ffn_dim_multiplier": 1,
            "attention_qkv_bias": False,
        },
        "quant": {
            "8da4w": {"qmode": "8da4w", "group_size": 32, "embedding_quantize": "4,32"},
            "8da8w": {"qmode": "8da8w", "group_size": 0, "embedding_quantize": "8,0"},
        },
    },
}

# What the HF->Meta converter needs to see in the snapshot.
_WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "config.json"]

# HunYuan's q/k norm names -> the Qwen3 names qwen3's converter maps.
_HY_TO_QWEN3 = {
    ".self_attn.query_layernorm.": ".self_attn.q_norm.",
    ".self_attn.key_layernorm.": ".self_attn.k_norm.",
}


def checkpoint(source_repo: str) -> str:
    """Downloads the upstream HF weights and converts them to Meta format.

    Cached under ~/.cache/exporter_meta_checkpoints, keyed by source repo, so
    each size converts once per machine and every backend shares it.
    """
    from pathlib import Path

    import torch
    from executorch.examples.models.qwen3.convert_weights import (
        load_checkpoint, qwen_3_tune_to_meta,
    )
    from huggingface_hub import snapshot_download

    cache = Path.home() / ".cache" / "exporter_meta_checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{source_repo.replace('/', '_')}.pth"
    if out.exists():
        return str(out)
    src = snapshot_download(source_repo, allow_patterns=_WEIGHT_PATTERNS)
    sd = load_checkpoint(src)
    for hy, qwen in _HY_TO_QWEN3.items():
        sd = {k.replace(hy, qwen): v for k, v in sd.items()}
    sd = qwen_3_tune_to_meta(sd)
    torch.save(sd, out)
    return str(out)


def model_schema(spec: dict, consts: dict) -> str:
    """The react-native-executorch contract baked as ``get_model_schema``.

    Without it RNE reads shapes from MethodMeta, which reports the token
    dimension as constant, and its text-LLM spec (a token range) rejects the
    model. Logits are float32 for every backend (MLX casts them, see
    mlx/export.py).
    """
    import json
    from pathlib import Path

    import jsonschema

    schema = {
        "forward": {
            "inputs": [
                {
                    "kind": "Tensor",
                    "dtype": "int64",
                    "shape": [
                        {"kind": "constant", "value": 1},
                        {
                            "kind": "range",
                            "range": {"min": 1, "max": consts["get_max_seq_len"] - 1, "step": 1},
                        },
                    ],
                },
                {"kind": "Tensor", "dtype": "int64", "shape": [{"kind": "constant", "value": 1}]},
            ],
            "outputs": [
                {
                    "kind": "Tensor",
                    "dtype": "float32",
                    "shape": [
                        {"kind": "constant", "value": 1},
                        {"kind": "constant", "value": spec["params"]["vocab_size"]},
                    ],
                }
            ],
            "runtimeConstraints": [],
        }
    }
    root = Path(__file__).resolve().parents[2] / "model.schema.json"
    jsonschema.validate(instance=schema, schema=json.loads(root.read_text()))
    return json.dumps(schema)
