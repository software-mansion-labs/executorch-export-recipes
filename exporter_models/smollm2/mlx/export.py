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

# Weight-only int8, not the 4w the other MLX LLMs use. SmolLM2 is too small
# to survive 4-bit: _params.py records that at 135m the int4 build answers a
# "reverse a string" prompt with nonsense re.sub code and at 360m it refuses
# ordinary questions, while int8 tracks bf16. The XNNPACK build ships int8
# for the same reason.
#
# NOTE: export_llama has no "8w" mode, so _quantize_with_8w below adds it: int8
# through torchao's IntxWeightOnlyConfig, the same path as its "4w". Plain
# qmode="int8" is the legacy handler and lowers to dequantize + dense AddmmNode,
# which measured 2x slower than XNNPACK 8da8w.
MLX_QMODE = "8w"


def _load_params():
    name = "smollm2_mlx_params"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _checkpoint(source_repo: str) -> str:
    """Downloads the upstream HF weights and converts them to Meta format."""
    _CKPT_CACHE.mkdir(parents=True, exist_ok=True)
    out = _CKPT_CACHE / f"{source_repo.replace('/', '_')}.pth"
    if out.exists():
        return str(out)
    # Imported only when converting: the converter pulls in torchtune.
    from executorch.examples.models.smollm2.convert_weights import convert_weights
    from huggingface_hub import snapshot_download

    src = snapshot_download(source_repo, allow_patterns=_WEIGHT_PATTERNS)
    # Some converters write scratch files into the cwd; keep them out of the repo.
    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        convert_weights(src, str(out))
    finally:
        os.chdir(cwd)
    return str(out)


def _quantize_with_8w(orig_quantize):
    """Wraps export_llama's quantize() with an "8w" mode mirroring its "4w"."""

    def quantize(model, qmode, *args, **kwargs):
        if qmode != "8w":
            return orig_quantize(model, qmode, *args, **kwargs)
        import torch
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

        group_size = kwargs.get("group_size") or 256
        hqq = kwargs.get("quantize_with_hqq", True)
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int8,
                granularity=PerGroup(group_size),
                intx_choose_qparams_algorithm="hqq_scale_only" if hqq else "affine",
            ),
        )
        return model

    return quantize


def export(config: dict, out_dir: str) -> str:
    from executorch.examples.models.llama.export_llama_lib import export_llama
    from executorch.extension.llm.export.config.llm_config import (
        DtypeOverride, LlmConfig, ModelType,
    )

    p = _load_params()
    size = config.get("size") or "135m"
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
    max_seq = consts["get_max_seq_len"]
    max_ctx = consts["get_max_context_len"]
    vocab_sz = consts.get("get_vocab_size", spec["params"]["vocab_size"])

    # The RNE text runner validates `forward` against this companion schema:
    # an explicit range for the token axis, a [1] position tensor, and rank-2
    # float32 logits.
    model_schema = {
        "forward": {
            "inputs": [
                {
                    "kind": "Tensor",
                    "dtype": "int64",
                    "shape": [
                        {"kind": "constant", "value": 1},
                        {"kind": "range", "range": {"min": 1, "max": max_seq - 1, "step": 1}},
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
                    "dtype": "float32",
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

        jsonschema.validate(instance=model_schema, schema=json.loads(root_schema_path.read_text()))

    metadata_dict = {k: consts[k] for k in ("get_bos_id", "get_eos_ids") if k in consts}
    metadata_dict["get_model_schema"] = json.dumps(model_schema)
    cfg.base.metadata = json.dumps(metadata_dict)
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    # bf16 throughout, with weight-only quantization on top. This mirrors
    # lfm2_5_text; the MLX backend takes bf16 activations, unlike the XNNPACK
    # path which overrides to fp32 for its quantized builds. Only the logits
    # are cast to float32, below.
    cfg.model.dtype_override = DtypeOverride.bf16
    cfg.quantization.qmode = MLX_QMODE
    cfg.quantization.group_size = MLX_GROUP_SIZE
    cfg.backend.mlx.enabled = True
    cfg.export.max_seq_length = max_seq
    cfg.export.max_context_length = max_ctx
    cfg.export.output_name = out_path

    # The runner only accepts float32 logits; cast the bf16 ones on the way out.
    import torch
    from executorch.examples.models.llama.llama_transformer import Transformer

    orig_forward = Transformer.forward

    def forward_fp32_logits(self, tokens=None, attn_options=None, *args, **kwargs):
        out = orig_forward(self, tokens=tokens, attn_options=attn_options, *args, **kwargs)
        if isinstance(out, torch.Tensor):
            return out.to(torch.float32)
        if isinstance(out, tuple):
            return tuple(
                x.to(torch.float32) if isinstance(x, torch.Tensor) and x.dtype.is_floating_point else x
                for x in out
            )
        return out

    Transformer.forward = forward_fp32_logits

    from executorch.examples.models.llama.source_transformation import quantize as quantize_mod

    orig_quantize = quantize_mod.quantize
    quantize_mod.quantize = _quantize_with_8w(orig_quantize)

    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        export_llama(cfg)
    finally:
        quantize_mod.quantize = orig_quantize
        Transformer.forward = orig_forward
        os.chdir(cwd)
        os.unlink(params_path)
    return out_path
