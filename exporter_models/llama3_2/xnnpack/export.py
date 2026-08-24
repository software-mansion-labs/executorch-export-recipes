import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_params():
    name = "llama3_2_xnnpack_params"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _weights(source_repo: str, precision: str) -> tuple[str, str]:
    """Fetch the native-format checkpoint; return (params.json, checkpoint).

    Both repos are gated -- `hf auth login` first. The plain Instruct repos keep
    the native files under `original/`; the SpinQuant repos keep them at the
    root.
    """
    from huggingface_hub import snapshot_download

    subdir = "" if precision == "spinquant" else "original"
    patterns = [f"{subdir}/*"] if subdir else ["consolidated.00.pth", "params.json"]
    root = Path(snapshot_download(source_repo, allow_patterns=patterns)) / subdir
    return str(root / "params.json"), str(root / "consolidated.00.pth")


def export(config: dict, out_dir: str) -> str:
    from executorch.examples.models.llama.export_llama_lib import export_llama
    from executorch.extension.llm.export.config.llm_config import (
        DtypeOverride, LlmConfig, ModelType, PreqMode, SpinQuant,
    )

    p = _load_params()
    size = config.get("size") or "1b"
    variant = config["variant"]
    precision = variant["precision"]

    out_path = str(Path(out_dir) / variant["file"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    params_path, checkpoint = _weights(p.SIZES[size][precision], precision)

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
    if precision == "bf16":
        cfg.model.dtype_override = DtypeOverride.bf16
    else:  # spinquant: the checkpoint is already quantized, just load it as such
        cfg.model.dtype_override = DtypeOverride.fp32
        cfg.base.preq_mode = PreqMode(p.SPINQUANT["preq_mode"])
        cfg.base.preq_group_size = p.SPINQUANT["preq_group_size"]
        cfg.base.preq_embedding_quantize = p.SPINQUANT["preq_embedding_quantize"]
        cfg.quantization.use_spin_quant = SpinQuant.native
    cfg.backend.xnnpack.enabled = True
    cfg.backend.xnnpack.extended_ops = True
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
