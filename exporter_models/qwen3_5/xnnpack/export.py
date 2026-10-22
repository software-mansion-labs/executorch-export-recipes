import importlib.util
import json
import os
import tempfile
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent.parent


def export(config: dict, out_dir: str) -> str:
    from executorch.examples.models.llama.export_llama_lib import export_llama
    from executorch.extension.llm.export.config.llm_config import (
        DtypeOverride,
        LlmConfig,
        ModelType,
    )

    spec = json.loads((_MODEL_DIR / "_params.json").read_text())[config["size"]]
    module_spec = importlib.util.spec_from_file_location(
        "qwen3_5_checkpoint", _MODEL_DIR / "_checkpoint.py"
    )
    support = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(support)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = config["variant"]
    cfg = LlmConfig()
    cfg.base.model_class = ModelType(spec["model_class"])
    cfg.base.checkpoint, source = support.resolve(spec)
    cfg.base.metadata = support.metadata(variant)
    support.write_assets(source, out_dir)
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = False
    cfg.model.enable_dynamic_shape = False
    cfg.model.dtype_override = DtypeOverride.fp32
    cfg.quantization.qmode = "8da4w"
    cfg.quantization.group_size = 64
    cfg.quantization.embedding_quantize = "4,64"
    cfg.backend.xnnpack.enabled = True
    cfg.backend.xnnpack.extended_ops = True
    cfg.export.max_seq_length = variant["constants"]["get_max_seq_len"]
    cfg.export.max_context_length = variant["constants"]["get_max_context_len"]
    cfg.export.output_name = str(out_dir / variant["file"])

    with tempfile.TemporaryDirectory(prefix="qwen3_5_export_") as temporary:
        params = Path(temporary) / "params.json"
        params.write_text(json.dumps(spec["params"]))
        cfg.base.params = str(params)
        cwd = os.getcwd()
        os.chdir(temporary)
        try:
            with support.rewindable_attention("xnnpack"):
                export_llama(cfg)
        finally:
            os.chdir(cwd)
    return cfg.export.output_name
