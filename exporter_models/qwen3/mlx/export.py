import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# Meta-format checkpoints converted from the upstream HF weights, keyed by
# source repo so each size converts once per machine. Shared with the xnnpack
# exporter, which writes the same files.
_CKPT_CACHE = Path.home() / ".cache" / "exporter_meta_checkpoints"

_WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "config.json"]

# Group size for MLX weight quantization, matching gemma4 and lfm2_5_text so the
# MLX builds are comparable across models.
MLX_GROUP_SIZE = 64

# 4w, matching gemma4 and lfm2_5_text.
MLX_QMODE = "4w"


def _load_params():
    name = "qwen3_mlx_params"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _checkpoint(source_repo: str) -> str:
    """Downloads the upstream HF weights and converts them to Meta format."""
    _CKPT_CACHE.mkdir(parents=True, exist_ok=True)
    out = _CKPT_CACHE / f"{source_repo.replace('/', '_')}.pth"
    if out.exists():
        return str(out)
    # Imported only when converting: the converter pulls in torchtune.
    from executorch.examples.models.qwen3.convert_weights import convert_weights
    from huggingface_hub import snapshot_download
    src = snapshot_download(source_repo, allow_patterns=_WEIGHT_PATTERNS)
    # Some converters write scratch files into the cwd; keep them out of the repo.
    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        convert_weights(src, str(out))
    finally:
        os.chdir(cwd)
    return str(out)


def export(config: dict, out_dir: str) -> str:
    from executorch.examples.models.llama.export_llama_lib import export_llama
    from executorch.extension.llm.export.config.llm_config import (
        DtypeOverride, LlmConfig, ModelType,
    )

    p = _load_params()
    size = config.get("size") or "0_6b"
    spec = p.SIZES[size] if hasattr(p, "SIZES") else p.PARAMS
    variant = config["variant"]

    # Resolve before the chdir below: export_llama writes output_name relative
    # to the cwd, so a relative --output-dir would land in the temp dir.
    out_dir = Path(out_dir).resolve()
    out_path = str(out_dir / variant["file"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec["params"], f)
        params_path = f.name

    cfg = LlmConfig()
    cfg.base.model_class = ModelType(spec["model_class"])
    cfg.base.params = params_path
    # Pass the checkpoint explicitly: export_llama's own auto-download maps
    # model_class -> HUGGING_FACE_REPO_IDS, which points at the base weights
    # rather than the Instruct ones we publish.
    cfg.base.checkpoint = _checkpoint(spec["source"])

    consts = variant.get("constants", {})
    max_seq = consts.get("get_max_seq_len", 2048)
    max_ctx = consts.get("get_max_context_len", 8192)
    vocab_sz = consts.get("get_vocab_size", spec["params"]["vocab_size"])
    out_dtype = "float32"

    model_schema = {
        "forward": {
            "inputs": [
                {
                    "kind": "Tensor",
                    "dtype": "int64",
                    "shape": [
                        {"kind": "constant", "value": 1},
                        {
                            "kind": "range",
                            "range": {
                                "min": 1,
                                "max": max_seq - 1,
                                "step": 1,
                            },
                        },
                    ],
                },
                {
                    "kind": "Tensor",
                    "dtype": "int64",
                    "shape": [{"kind": "constant", "value": 1}],
                },
            ],
            "outputs": [
                {
                    "kind": "Tensor",
                    "dtype": out_dtype,
                    "shape": [
                        {"kind": "constant", "value": 1},
                        {"kind": "constant", "value": vocab_sz},
                    ],
                }
            ],
            "runtimeConstraints": [],
        }
    }
    root_schema_path = _MODEL_DIR.parent / "model.schema.json"
    if root_schema_path.exists():
        import jsonschema

        root_schema = json.loads(root_schema_path.read_text())
        jsonschema.validate(instance=model_schema, schema=root_schema)

    metadata_dict = {k: consts[k] for k in ("get_bos_id", "get_eos_ids") if k in consts}
    metadata_dict["get_model_schema"] = json.dumps(model_schema)
    cfg.base.metadata = json.dumps(metadata_dict)
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    cfg.model.dtype_override = DtypeOverride.bf16
    cfg.quantization.qmode = MLX_QMODE
    cfg.quantization.group_size = MLX_GROUP_SIZE
    cfg.backend.mlx.enabled = True
    cfg.export.max_seq_length = max_seq
    cfg.export.max_context_length = max_ctx
    cfg.export.output_name = out_path

    # Patch Transformer.forward to cast output logits to float32
    import torch
    from executorch.examples.models.llama.llama_transformer import Transformer

    orig_forward = Transformer.forward

    def patched_forward(self, tokens=None, attn_options=None, *args, **kwargs):
        out = orig_forward(self, tokens=tokens, attn_options=attn_options, *args, **kwargs)
        if isinstance(out, torch.Tensor):
            return out.to(torch.float32)
        elif isinstance(out, tuple):
            return tuple(
                x.to(torch.float32) if isinstance(x, torch.Tensor) and x.dtype.is_floating_point else x
                for x in out
            )
        return out

    Transformer.forward = patched_forward

    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        export_llama(cfg)
    finally:
        Transformer.forward = orig_forward
        os.chdir(cwd)
        os.unlink(params_path)
    return out_path

