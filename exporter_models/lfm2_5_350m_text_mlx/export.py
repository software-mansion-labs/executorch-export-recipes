import json
import os
import tempfile
from pathlib import Path

_LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv", "conv", "full_attention",
    "conv", "conv", "full_attention", "conv", "full_attention", "conv",
    "full_attention", "conv", "full_attention", "conv",
]

_PARAMS = {
    "dim": 1024, "ffn_dim_multiplier": 1, "hidden_dim": 4608,
    "n_heads": 16, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5,
    "rope_theta": 1000000.0, "use_scaled_rope": False,
    "vocab_size": 65536, "use_hf_rope": True, "use_qk_norm": True,
    "qk_norm_before_rope": True, "layer_types": _LAYER_TYPES,
}

GROUP_SIZE = 64
PREFILL_CAP = 64
CONTEXT_LEN = 2048


def export(config: dict, out_dir: str) -> str:
    from executorch.examples.models.llama.export_llama_lib import export_llama
    from executorch.extension.llm.export.config.llm_config import (
        DtypeOverride, LlmConfig, ModelType,
    )

    out_path = str(Path(out_dir) / config["variant"]["file"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(_PARAMS, f)
        params_path = f.name

    cfg = LlmConfig()
    cfg.base.model_class = ModelType("lfm2_5_350m")
    cfg.base.params = params_path
    cfg.base.metadata = '{"get_bos_id": 1, "get_eos_ids":[7]}'
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    cfg.model.dtype_override = DtypeOverride.bf16
    cfg.quantization.qmode = "4w"
    cfg.quantization.group_size = GROUP_SIZE
    cfg.backend.mlx.enabled = True
    cfg.export.max_seq_length = PREFILL_CAP + 1
    cfg.export.max_context_length = CONTEXT_LEN
    cfg.export.output_name = out_path

    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        export_llama(cfg)
    finally:
        os.chdir(cwd)
        os.unlink(params_path)
    return out_path
