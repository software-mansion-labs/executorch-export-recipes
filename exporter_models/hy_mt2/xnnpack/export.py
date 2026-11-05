import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_params():
    name = "hy_mt2_xnnpack_params"
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
    size = config.get("size") or "1_8b"
    spec = p.SIZES[size]
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
    # model_class -> HUGGING_FACE_REPO_IDS, which knows nothing of Hy-MT2.
    cfg.base.checkpoint = p.checkpoint(spec["source"])

    # Runner constants come from the config -- the single source of truth.
    # export_llama derives the rest (vocab) from params.
    consts = variant.get("constants", {})
    metadata = {k: consts[k] for k in ("get_bos_id", "get_eos_ids") if k in consts}
    metadata["get_model_schema"] = p.model_schema(spec, consts)
    cfg.base.metadata = json.dumps(metadata)
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    if variant["precision"] == "bf16":
        cfg.model.dtype_override = DtypeOverride.bf16
    else:  # 8da4w, 8da8w
        cfg.model.dtype_override = DtypeOverride.fp32
        # The recipe names its qmode; the rest of the dict are quantization
        # knobs. An unset knob means "executorch default" (see _params.py).
        knobs = dict(spec["quant"][variant["precision"]])
        cfg.quantization.qmode = knobs.pop("qmode", "8da4w")
        for knob, value in knobs.items():
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
