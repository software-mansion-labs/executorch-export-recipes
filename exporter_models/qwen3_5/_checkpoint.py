"""Pinned text weights and tokenizer assets from the Qwen3.5 checkpoints."""

import json
import shutil
from pathlib import Path


def resolve(spec: dict) -> tuple[str, Path]:
    from executorch.examples.models.qwen3_5.convert_weights import convert_weights
    from huggingface_hub import snapshot_download

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
                "chat_template.jinja",
                "LICENSE*",
            ],
        )
    )
    upstream = json.loads((source / "config.json").read_text())["text_config"]
    for key, expected in {
        "hidden_size": spec["params"]["dim"],
        "intermediate_size": spec["params"]["hidden_dim"],
        "num_hidden_layers": spec["params"]["n_layers"],
        "vocab_size": spec["params"]["vocab_size"],
        "layer_types": spec["params"]["layer_types"],
    }.items():
        if upstream[key] != expected:
            raise ValueError(f"{spec['source']}: unexpected {key}: {upstream[key]!r}")
    cache = Path.home() / ".cache" / "exporter_meta_checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / f"{spec['source'].replace('/', '_')}_{spec['revision']}.pth"
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix(".tmp")
        convert_weights(str(source), str(temporary))
        temporary.replace(checkpoint)
    return str(checkpoint), source


def metadata(variant: dict) -> str:
    constants = dict(variant["constants"])
    schema = {}
    for name, method in variant["methods"].items():
        converted = {}
        for side in ("inputs", "outputs"):
            converted[side] = []
            for index, tensor in enumerate(method[side]):
                shape = [{"kind": "constant", "value": d} for d in tensor["shape"]]
                if name == "forward" and side == "inputs" and index == 0:
                    # The text runner accepts a range; this static model supports one token.
                    shape[1] = {
                        "kind": "range",
                        "range": {"min": 1, "max": 1, "step": 1},
                    }
                converted[side].append(
                    {"kind": "Tensor", "dtype": tensor["dtype"], "shape": shape}
                )
        schema[name] = {**converted, "runtimeConstraints": []}
    constants["get_model_schema"] = json.dumps(schema)
    return json.dumps(constants)


def write_assets(source: Path, out_dir: Path) -> None:
    shutil.copyfile(source / "tokenizer.json", out_dir / "tokenizer.json")
    config = json.loads((source / "tokenizer_config.json").read_text())
    template = (source / "chat_template.jinja").read_text()
    # Strip thinking from every assistant message so appending a user keeps the prefix stable.
    template = template.replace("if loop.index0 > ns.last_query_index", "if false")
    # Session initialization may render only the system message, before any user query.
    template = template.replace(
        "{%- if ns.multi_step_tool %}\n    {{- raise_exception('No user query found in messages.') }}\n{%- endif %}",
        "",
    )
    config["chat_template"] = template
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    shutil.copyfile(source / "LICENSE", out_dir / "LICENSE")


def rewindable_attention(backend):
    import importlib.util
    import sys

    name = "qwen3_5_rewind_attention"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("_attention.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].install(backend)
