"""Pinned, sharded checkpoints and tokenizer assets for LFM2.5 2.6B."""

import json
import os
import shutil
from pathlib import Path


def resolve(spec: dict, size: str) -> tuple[str, Path]:
    from executorch.examples.models.lfm2.convert_weights import lfm_2_to_meta
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch

    source_root = os.environ.get("LFM2_5_SOURCE_ROOT")
    source = Path(
        snapshot_download(
            spec["source"],
            revision=spec["revision"],
            allow_patterns=[
                "*.safetensors",
                "*.safetensors.index.json",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
                "chat_template.jinja",
                "LICENSE*",
            ],
            local_dir=str(Path(source_root) / size) if source_root else None,
        )
    )
    upstream = json.loads((source / "config.json").read_text())
    expected = spec["params"]
    for key, value in {
        "hidden_size": expected["dim"],
        "intermediate_size": expected["hidden_dim"],
        "num_hidden_layers": expected["n_layers"],
        "vocab_size": expected["vocab_size"],
        "num_attention_heads": expected["n_heads"],
        "num_key_value_heads": expected["n_kv_heads"],
        "layer_types": expected["layer_types"],
        "conv_L_cache": 3,
        "conv_bias": False,
    }.items():
        if upstream[key] != value:
            raise ValueError(
                f"{spec['source']}: unexpected {key}: {upstream[key]!r} != {value!r}"
            )
    rope_theta = upstream.get("rope_parameters", {}).get(
        "rope_theta", upstream.get("rope_theta")
    )
    if rope_theta != expected["rope_theta"]:
        raise ValueError(f"Unexpected RoPE theta: {rope_theta}")

    cache = Path.home() / ".cache" / "exporter_meta_checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / f"{spec['source'].replace('/', '_')}_{spec['revision']}.pth"
    if not checkpoint.exists():
        index_path = source / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            shards = sorted(set(index["weight_map"].values()))
        else:
            shards = ["model.safetensors"]
        state = {}
        for shard in shards:
            weights = load_file(str(source / shard))
            if state.keys() & weights.keys():
                raise ValueError(f"Duplicate weight names in shard {shard}")
            state.update(weights)
        converted = lfm_2_to_meta(state)
        if "lm_head.weight" in converted:
            converted["output.weight"] = converted.pop("lm_head.weight")
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(converted, temporary)
        temporary.replace(checkpoint)
    return str(checkpoint), source


def metadata(variant: dict) -> str:
    """Expose the dynamic prefill bound to RNE's schema-aware text runner."""
    constants = dict(variant["constants"])
    methods = variant["methods"]
    schema = {}
    for name, method in methods.items():
        converted = {}
        for side in ("inputs", "outputs"):
            converted[side] = []
            for i, tensor in enumerate(method[side]):
                shape = [{"kind": "constant", "value": d} for d in tensor["shape"]]
                if name == "forward" and side == "inputs" and i == 0:
                    shape[1] = {
                        "kind": "range",
                        "range": {"min": 1, "max": tensor["shape"][1], "step": 1},
                    }
                converted[side].append(
                    {"kind": "Tensor", "dtype": tensor["dtype"], "shape": shape}
                )
        schema[name] = {**converted, "runtimeConstraints": []}
    # Constant methods use their compiled metadata. ExecuTorch represents a
    # list constant such as get_eos_ids as separate scalar outputs.
    constants["get_model_schema"] = json.dumps(schema)
    return json.dumps(constants)


def write_assets(source: Path, out_dir: Path, size: str) -> None:
    shutil.copyfile(source / "tokenizer.json", out_dir / "tokenizer.json")
    config = json.loads((source / "tokenizer_config.json").read_text())
    template = source / "chat_template.jinja"
    if not config.get("chat_template") and template.exists():
        config["chat_template"] = template.read_text()
    if not config.get("chat_template"):
        raise ValueError(f"No chat template available for {size}")
    if size == "2_6b":
        # Strip thinking from every assistant message, like the Qwen3 template.
        # Appending a user message must not change previously rendered content.
        config["chat_template"] = config["chat_template"].replace(
            "preserve_thinking or loop.index0 > ns.last_user_index",
            "preserve_thinking",
        )
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    for name in ("generation_config.json", "LICENSE"):
        if (source / name).exists():
            shutil.copyfile(source / name, out_dir / name)
