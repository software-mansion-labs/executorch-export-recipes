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
    name = "hammer2_1_xnnpack_params"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _checkpoint(source_repo: str) -> str:
    """Download the upstream HF weights and convert them to Meta format."""
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
    # model_class -> HUGGING_FACE_REPO_IDS, which points at the base weights
    # rather than the Instruct ones we publish.
    cfg.base.checkpoint = _checkpoint(spec["source"])

    # Runner constants come from the config -- the single source of truth.
    # export_llama derives the rest (vocab) from params.
    consts = variant.get("constants", {})
    max_seq = consts.get("get_max_seq_len", 2048)
    vocab_sz = consts.get("get_vocab_size", spec["params"]["vocab_size"])
    out_dtype = "float32" if variant["precision"] != "fp16" and variant["precision"] != "bf16" else ("float16" if variant["precision"] == "fp16" else "bfloat16")

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
    if variant["precision"] == "bf16":
        cfg.model.dtype_override = DtypeOverride.bf16
    else:  # 8da4w
        cfg.model.dtype_override = DtypeOverride.fp32
        # A recipe may name its own qmode (e.g. "torchao:8da8w" for int8
        # weights); the rest of the dict are quantization knobs. An unset knob
        # means "executorch default" (see _params.py).
        knobs = dict(spec["quant"])
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
