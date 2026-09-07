import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# Group size for MLX weight quantization, matching gemma4 and lfm2_5_text so the
# MLX builds are comparable across models.
MLX_GROUP_SIZE = 64

# 4w, matching gemma4 and lfm2_5_text.
MLX_QMODE = "4w"


def _load_params():
    name = "llama3_2_mlx_params"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _weights(source_repo: str) -> tuple[str, str]:
    """Fetches the native-format checkpoint; returns (params.json, checkpoint).

    The repo is gated -- `hf auth login` first. Only the bf16 Instruct repo is
    used here: MLX quantizes from bf16 weights, so the SpinQuant checkpoints the
    XNNPACK exporter can also take are not applicable.
    """
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(source_repo, allow_patterns=["original/*"])) / "original"
    return str(root / "params.json"), str(root / "consolidated.00.pth")


def export(config: dict, out_dir: str) -> str:
    from executorch.examples.models.llama.export_llama_lib import export_llama
    from executorch.extension.llm.export.config.llm_config import (
        DtypeOverride, LlmConfig, ModelType,
    )

    p = _load_params()
    size = config.get("size") or "1b"
    variant = config["variant"]

    # Resolve before the chdir below: export_llama writes output_name relative
    # to the cwd, so a relative --output-dir would land in the temp dir.
    out_dir = Path(out_dir).resolve()
    out_path = str(out_dir / variant["file"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always the bf16 repo, whatever the variant is called: the MLX build
    # quantizes bf16 weights itself rather than loading a pre-quantized one.
    params_path, checkpoint = _weights(p.SIZES[size]["bf16"])

    cfg = LlmConfig()
    cfg.base.model_class = ModelType(p.MODEL_CLASS)
    cfg.base.params = params_path
    cfg.base.checkpoint = checkpoint

    # Runner constants come from the config -- the single source of truth.
    consts = variant.get("constants", {})
    cfg.base.metadata = json.dumps(
        {k: consts[k] for k in ("get_bos_id", "get_eos_ids") if k in consts}
    )
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    # bf16 throughout with weight-only quantization on top, mirroring
    # lfm2_5_text. The MLX backend takes bf16 activations, unlike the XNNPACK
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
    return out_path
