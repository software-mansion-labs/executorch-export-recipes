"""Hammer2.1 MLX export.

Hammer2.1 is a Qwen2.5 fine-tune: it borrows qwen2_5 model classes and the
qwen2_5 HF->Meta weight converter (see _params.py), so this is the qwen2_5
MLX recipe pointed at Hammer's params and source repos.
"""
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
    name = "hammer2_1_mlx_params"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _checkpoint(source_repo: str) -> str:
    """Downloads the upstream HF weights and converts them to Meta format."""
    from executorch.examples.models.qwen2_5.convert_weights import convert_weights
    from huggingface_hub import snapshot_download

    _CKPT_CACHE.mkdir(parents=True, exist_ok=True)
    out = _CKPT_CACHE / f"{source_repo.replace('/', '_')}.pth"
    if out.exists():
        return str(out)
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
    size = config.get("size") or "0_5b"
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
    cfg.base.metadata = json.dumps(
        {k: consts[k] for k in ("get_bos_id", "get_eos_ids") if k in consts}
    )
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    # bf16 throughout, with weight-only quantization on top. This mirrors
    # lfm2_5_text; the MLX backend takes bf16 activations, unlike the XNNPACK
    # path which overrides to fp32 for its quantized builds.
    cfg.model.dtype_override = DtypeOverride.bf16
    cfg.quantization.qmode = MLX_QMODE
    cfg.quantization.group_size = MLX_GROUP_SIZE
    cfg.backend.mlx.enabled = True
    cfg.export.max_seq_length = consts["get_max_seq_len"]
    cfg.export.max_context_length = consts["get_max_context_len"]
    cfg.export.output_name = out_path

    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        export_llama(cfg)
    finally:
        os.chdir(cwd)
        os.unlink(params_path)
    return out_path
