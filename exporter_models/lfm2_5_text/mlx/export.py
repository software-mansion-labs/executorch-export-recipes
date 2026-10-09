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

    # Resolve before the chdir below: export_llama writes output_name relative
    # to the cwd, so a relative --output-dir would land in the temp dir.
    out_dir = Path(out_dir).resolve()
    out_path = str(out_dir / config["variant"]["file"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec["params"], f)
        params_path = f.name

    cfg = LlmConfig()
    cfg.base.model_class = ModelType(spec["model_class"])
    cfg.base.params = params_path
    # Runner constants come from the config -- the single source of truth.
    # export_llama derives the rest (vocab) from params.
    consts = config["variant"].get("constants", {})
    max_ctx = consts.get("get_max_context_len", 16384)
    max_seq = consts.get("get_max_seq_len", 2048)
    vocab_sz = consts.get("get_vocab_size", spec["params"]["vocab_size"])
    out_dtype = "float32"

    # Generate and validate get_model_schema matching model.schema.json
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
    root_schema_path = _MODEL_DIR.parent.parent / "model.schema.json"
    if root_schema_path.exists():
        import jsonschema

        root_schema = json.loads(root_schema_path.read_text())
        jsonschema.validate(instance=model_schema, schema=root_schema)

    metadata_dict = dict(consts)
    metadata_dict["get_model_schema"] = json.dumps(model_schema)
    cfg.base.metadata = json.dumps(metadata_dict)
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    cfg.model.dtype_override = DtypeOverride.bf16
    cfg.quantization.group_size = p.MLX_GROUP_SIZE
    cfg.quantization.qmode = "4w"
    cfg.backend.mlx.enabled = True
    cfg.export.max_seq_length = consts["get_max_seq_len"]
    cfg.export.max_context_length = consts["get_max_context_len"]
    cfg.export.output_name = out_path

    # Patch Transformer.forward during export to cast output logits to float32 without weight bloat
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
