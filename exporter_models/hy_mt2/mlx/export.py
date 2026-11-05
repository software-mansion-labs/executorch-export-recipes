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

# Weight-only qmode per variant precision (NOTES.md).
MLX_QMODES = {"int8": "8w", "int4": "4w"}


def _load_params():
    name = "hy_mt2_mlx_params"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch_weight_only() -> None:
    """Teaches export_llama ``8w`` and reroutes ``4w``: weight-only, PerGroup.

    export_llama's weight-only torchao path is hardcoded to int4 (``4w``) and
    quantizes only the linears. Both branches here are IntxWeightOnlyConfig
    applied to the embedding too (MLX lowers a torchao-quantized embedding to a
    quantized gather), which at a 120k vocab is ~0.5 GB of bf16 otherwise. Any
    other qmode falls through to the stock function.
    """
    import torch
    from executorch.examples.models.llama.source_transformation import quantize as q

    if getattr(q.quantize, "_hy_mt2_wo", False):
        return
    stock = q.quantize

    def quantize(model, qmode, *args, group_size=None, **kwargs):
        bits = {"8w": torch.int8, "4w": torch.int4}.get(qmode)
        if bits is None:
            return stock(model, qmode, *args, group_size=group_size, **kwargs)
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

        config = IntxWeightOnlyConfig(weight_dtype=bits, granularity=PerGroup(group_size))
        quantize_(model, config, filter_fn=lambda m, _: isinstance(m, (torch.nn.Linear, torch.nn.Embedding)))
        return model

    quantize._hy_mt2_wo = True
    q.quantize = quantize


def _patch_fp32_logits() -> None:
    """Makes ``forward`` return float32 logits, as the other MLX LLMs do.

    With ``dtype_override`` bf16 the output head emits bf16 logits, which the
    react-native-executorch LLM runner (float32 logits for its sampler) rejects
    with "Unsupported ScalarType". The cast costs one [1, vocab] AsType.
    """
    from executorch.examples.models.llama import llama_transformer as lt

    if getattr(lt.Transformer.forward, "_hy_mt2_fp32", False):
        return
    stock = lt.Transformer.forward

    # Same parameters as the stock forward: torch.export binds dynamic shapes
    # by signature.
    def forward(self, tokens=None, attn_options=None, h=None):
        out = stock(self, tokens, attn_options, h)
        if isinstance(out, tuple):
            return (out[0].float(), *out[1:])
        return out.float()

    forward._hy_mt2_fp32 = True
    lt.Transformer.forward = forward


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

    consts = variant.get("constants", {})
    metadata = {k: consts[k] for k in ("get_bos_id", "get_eos_ids") if k in consts}
    metadata["get_model_schema"] = p.model_schema(spec, consts)
    cfg.base.metadata = json.dumps(metadata)
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    # bf16 throughout, with weight-only quantization on top. This mirrors
    # lfm2_5_text; the MLX backend takes bf16 activations, unlike the XNNPACK
    # path which overrides to fp32 for its quantized builds.
    cfg.model.dtype_override = DtypeOverride.bf16
    # Set after construction: LlmConfig validates qmode only in __post_init__,
    # and its list has no "8w" (see _patch_weight_only).
    _patch_weight_only()
    _patch_fp32_logits()
    cfg.quantization.qmode = MLX_QMODES[variant["precision"]]
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
        os.unlink(params_path)
    return out_path
