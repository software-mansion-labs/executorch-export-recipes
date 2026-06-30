import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_params():
    name = "lfm2_5_text_mlx_params"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.examples.models.llama.export_llama_lib import export_llama
    from executorch.extension.llm.export.config.llm_config import (
        DtypeOverride, LlmConfig, ModelType,
    )

    p = _load_params()
    size = config.get("size") or "350m"
    spec = p.SIZES[size]

    out_path = str(Path(out_dir) / config["variant"]["file"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec["params"], f)
        params_path = f.name

    cfg = LlmConfig()
    cfg.base.model_class = ModelType(spec["model_class"])
    cfg.base.params = params_path
    cfg.base.metadata = p.METADATA
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    cfg.model.dtype_override = DtypeOverride.bf16
    cfg.quantization.group_size = p.GROUP_SIZE
    cfg.quantization.qmode = "4w"
    cfg.backend.mlx.enabled = True
    cfg.export.max_seq_length = p.PREFILL_CAP + 1
    cfg.export.max_context_length = p.CONTEXT_LEN
    cfg.export.output_name = out_path

    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        export_llama(cfg)
    finally:
        os.chdir(cwd)
        os.unlink(params_path)
    return out_path
