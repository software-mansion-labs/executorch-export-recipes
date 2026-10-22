"""Regenerate every {backend}/{size}/config.json from one table.

The pairs share one architecture, so their configs differ only in names and
vocab size; keeping them generated stops the 24 files drifting apart.
Run: python exporter_models/opus_mt/_gen_configs.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "https://huggingface.co/software-mansion/react-native-executorch-spec/resolve/main/config.schema.json"
# vocab_size of each upstream checkpoint (config.json).
VOCAB = {"en_de": 58101, "de_en": 58101, "en_es": 65001, "es_en": 65001, "en_fr": 59514, "fr_en": 59514}
LICENSE = {"en_de": "cc-by-4.0"}
N_LAYERS, N_HEADS, HEAD_DIM = 6, 8, 64

# backend -> (source length bound, static?, [(precision, quantized, extra)])
BACKENDS = {
    "xnnpack": (256, False, [("fp32", False, {}),
                             ("int8", True, {"quantize": "8da8w-pc", "embedding_quantize": "int8"})]),
    "coreml": (128, True, [("fp16", False, {})]),
    "mlx": (256, False, [("bf16", False, {}), ("int8", True, {"quantize": "8w"})]),
    "vulkan": (256, False, [("fp16", False, {})]),
}


def methods(size: str, src: int) -> dict:
    kv = [{"shape": [N_HEADS, src, HEAD_DIM], "dtype": "float32"}] * (2 * N_LAYERS)
    ids = {"shape": [1, src], "dtype": "int64"}
    return {
        "encode": {"inputs": [ids, ids], "outputs": kv},
        "decode": {"inputs": [{"shape": [1, 1], "dtype": "int64"}, {"shape": [1], "dtype": "int64"}, ids] + kv,
                   "outputs": [{"shape": [1, 1, VOCAB[size]], "dtype": "float32"}]},
    }


for backend, (src, _static, variants) in BACKENDS.items():
    for size in VOCAB:
        cfg = {
            "$schema": SCHEMA, "model": "opus_mt", "family": "marian", "size": size,
            "capabilities": ["translation"], "backend": backend,
            "license": LICENSE.get(size, "apache-2.0"),
            "tokenizer": "tokenizer.json",
            "variants": [
                {"file": f"opus_mt_{size}_{backend}_{p}.pte", "precision": p, "quantized": q, **extra,
                 # One default per quantized / non-quantized group: the first listed.
                 "default": all(q2 != q for _, q2, _ in variants[:i]), "methods": methods(size, src)}
                for i, (p, q, extra) in enumerate(variants)
            ],
        }
        out = HERE / backend / size / "config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cfg, indent=2) + "\n")
