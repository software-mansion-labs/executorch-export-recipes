"""MiniCPM5's pinned Llama checkpoint and shared backend export setup."""

import json
import os
import tempfile
from pathlib import Path

import torch

SOURCE = "openbmb/MiniCPM5-2B"
REVISION = "f97400052a43d642bbc6e9975e2397e3ae6a6b52"
PARAMS = {
    "dim": 2048,
    "hidden_dim": 6144,
    "n_layers": 42,
    "n_heads": 16,
    "n_kv_heads": 2,
    "head_dim": 128,
    "norm_eps": 1e-6,
    "rope_theta": 5000000.0,
    "use_hf_rope": True,
    "use_scaled_rope": False,
    "vocab_size": 130560,
}


def resolve() -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    source = Path(
        snapshot_download(
            SOURCE,
            revision=REVISION,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "chat_template.jinja",
                "LICENSE*",
            ],
        )
    )
    cache = Path.home() / ".cache" / "exporter_meta_checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / f"openbmb_MiniCPM5-2B_{REVISION}.pth"
    if not checkpoint.exists():
        mapping = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }
        layers = {
            "self_attn.q_proj": "attention.wq",
            "self_attn.k_proj": "attention.wk",
            "self_attn.v_proj": "attention.wv",
            "self_attn.o_proj": "attention.wo",
            "mlp.gate_proj": "feed_forward.w1",
            "mlp.down_proj": "feed_forward.w2",
            "mlp.up_proj": "feed_forward.w3",
            "input_layernorm": "attention_norm",
            "post_attention_layernorm": "ffn_norm",
        }
        for i in range(PARAMS["n_layers"]):
            mapping.update(
                {
                    f"model.layers.{i}.{key}.weight": f"layers.{i}.{value}.weight"
                    for key, value in layers.items()
                }
            )
        state = {}
        for shard in sorted(source.glob("*.safetensors")):
            for key, value in load_file(str(shard)).items():
                state[mapping[key]] = value
        if set(state) != set(mapping.values()):
            raise ValueError(
                "MiniCPM5 checkpoint does not match the Llama architecture"
            )
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint)
    return checkpoint, source


def metadata(variant: dict) -> str:
    constants = dict(variant["constants"])
    schema = {}
    for name, method in variant["methods"].items():
        signature = {}
        for side in ("inputs", "outputs"):
            signature[side] = []
            for tensor in method[side]:
                shape = [
                    {"kind": "range", "range": {**d, "step": 1}}
                    if isinstance(d, dict)
                    else {"kind": "constant", "value": d}
                    for d in tensor["shape"]
                ]
                signature[side].append(
                    {"kind": "Tensor", "dtype": tensor["dtype"], "shape": shape}
                )
        schema[name] = {**signature, "runtimeConstraints": []}
    constants["get_model_schema"] = json.dumps(schema)
    return json.dumps(constants)


def write_assets(source: Path, out_dir: Path) -> None:
    tokenizer = json.loads((source / "tokenizer.json").read_text())
    stages = tokenizer["pre_tokenizer"]["pretokenizers"]
    # Fuse digit grouping into the word split: C++ Split drops unmatched spans.
    if stages[0]["pattern"]["Regex"] != r"\p{N}{1,3}":
        raise ValueError("Unexpected MiniCPM5 tokenizer digit split")
    stages[1]["pattern"]["Regex"] = (
        stages[1]["pattern"]["Regex"]
        .replace(r"\p{N}+", r"\p{N}{1,3}")
        .replace(r"\s+(?!\S)", r"\s+(?=\p{N})|\s+(?!\S)")
    )
    tokenizer["pre_tokenizer"]["pretokenizers"] = stages[1:]
    (out_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer, ensure_ascii=False) + "\n"
    )
    config = json.loads((source / "tokenizer_config.json").read_text())
    template = (source / "chat_template.jinja").read_text()
    # Strip reasoning from every assistant message so history stays monotonic.
    start = template.index("        {%- set reasoning_content = '' %}")
    end = template.index("        {%- if message.tool_calls %}", start)
    template = (
        template[:start]
        + (
            "        {%- if '</think>' in content %}\n"
            "            {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
            "        {%- endif %}\n"
            "        {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        )
        + template[end:]
    )
    template = template[: template.index("{%- if add_generation_prompt %}")] + (
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n<think>\\n' }}"
        "{%- endif %}"
    )
    config["chat_template"] = template
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )


def export(config: dict, out_dir: str) -> str:
    source_root = os.environ.get("MINICPM5_EXECUTORCH_ROOT")
    if source_root:
        import executorch.backends
        import executorch.examples.models.llama

        source_root = Path(source_root).resolve()
        executorch.backends.__path__.insert(0, str(source_root / "backends"))
        executorch.examples.models.llama.__path__.insert(
            0, str(source_root / "examples/models/llama")
        )

    from executorch.examples.models.llama.export_llama_lib import export_llama
    from executorch.extension.llm.export.config.llm_config import (
        DtypeOverride,
        LlmConfig,
        ModelType,
    )

    variant = config["variant"]
    backend = config["backend"]
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, source = resolve()
    write_assets(source, out_dir)
    cfg = LlmConfig()
    cfg.base.model_class = ModelType.llama3
    cfg.base.checkpoint = str(checkpoint)
    cfg.base.metadata = metadata(variant)
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    cfg.model.enable_dynamic_shape = True
    cfg.model.dtype_override = DtypeOverride.fp32
    cfg.quantization.qmode = "4w" if backend == "mlx" else "8da4w"
    cfg.quantization.group_size = 64
    cfg.quantization.embedding_quantize = "4,64"
    if backend == "mlx":
        cfg.backend.mlx.enabled = True
    else:
        cfg.backend.xnnpack.enabled = True
        cfg.backend.xnnpack.extended_ops = True
    cfg.export.max_seq_length = variant["constants"]["get_max_seq_len"]
    cfg.export.max_context_length = variant["constants"]["get_max_context_len"]
    cfg.export.output_name = str(out_dir / variant["file"])
    with tempfile.TemporaryDirectory(prefix="minicpm5_export_") as temporary:
        params_path = Path(temporary) / "params.json"
        params_path.write_text(json.dumps(PARAMS))
        cfg.base.params = str(params_path)
        cwd = os.getcwd()
        os.chdir(temporary)
        try:
            export_llama(cfg)
        finally:
            os.chdir(cwd)
    return cfg.export.output_name
