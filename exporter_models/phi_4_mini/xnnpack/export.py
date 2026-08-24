import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# Meta-format checkpoints converted from the upstream HF weights, keyed by
# source repo so each size converts once per machine.
_CKPT_CACHE = Path.home() / ".cache" / "exporter_meta_checkpoints"

# What the HF->Meta converter needs to see in the snapshot.
_WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "config.json"]


def _load_params():
    name = "phi_4_mini_xnnpack_params"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _checkpoint(source_repo: str) -> str:
    """Download the upstream HF weights and convert them to Meta format."""
    from executorch.examples.models.phi_4_mini.convert_weights import convert_weights
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
    variant = config["variant"]

    out_path = str(Path(out_dir) / variant["file"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(p.PARAMS, f)
        params_path = f.name

    cfg = LlmConfig()
    cfg.base.model_class = ModelType(p.MODEL_CLASS)
    cfg.base.params = params_path
    # Pass the checkpoint explicitly: export_llama's own auto-download maps
    # model_class -> HUGGING_FACE_REPO_IDS, which points at the base weights
    # rather than the Instruct ones we publish.
    cfg.base.checkpoint = _checkpoint(p.SOURCE)

    # Runner constants come from the config -- the single source of truth.
    # export_llama derives the rest (vocab) from params.
    consts = variant.get("constants", {})
    cfg.base.metadata = json.dumps(
        {k: consts[k] for k in ("get_bos_id", "get_eos_ids") if k in consts}
    )
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    if variant["precision"] == "bf16":
        cfg.model.dtype_override = DtypeOverride.bf16
    else:  # 8da4w
        cfg.model.dtype_override = DtypeOverride.fp32
        cfg.quantization.qmode = "8da4w"
        # An unset knob means "executorch default" (see _params.py).
        for knob, value in p.QUANT.items():
            if value is not None:
                setattr(cfg.quantization, knob, value)
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
        os.unlink(params_path)
    return out_path
