"""Supertonic 3 -> ExecuTorch XNNPACK (multi_pte).

The four sub-models are lowered as four separate .pte files (mirroring the ONNX
layout) so the host drives the same pipeline (text preprocessing, noise
sampling, the flow-matching loop) around them:

    duration_predictor(text_ids, style_dp,  text_mask)               -> duration
    text_encoder      (text_ids, style_ttl, text_mask)               -> text_emb
    vector_estimator  (noisy_latent, text_emb, style_ttl, text_mask,
                       latent_mask, current_step, total_step)         -> denoised_latent
    vocoder           (latent)                                        -> wav

Each .pte also carries a constant method ``get_model_schema`` advertising
its tensor parameters and dynamic bounds.

Weights: the upstream Supertone/supertonic-3 ONNX files, fetched from HF and
converted to PyTorch state dicts by the vendored ``_supertonic`` package.
"""

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# Upstream ONNX weights (openrail). convert.load_models extracts the four
# sub-model state dicts from these.
_ONNX_REPO = "Supertone/supertonic-3"
_ONNX_FILES = [
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
]

# Sub-model dynamic-shape bounds (min, max) for the text length T and latent
# length L, matching the shipped spec.
_T_MIN, _T_MAX = 8, 512
_L_MIN, _L_MAX = 4, 1000

# Static feature sizes (see build_example_inputs).
_N_STYLE = 50
_STYLE_DIM = 256
_DP_STYLE = (8, 16)
_TEXT_EMB_DIM = 256
_LATENT_DIM = 144  # ldim(24) * chunk_compress_factor(6)


def _load_convert():
    """Load the vendored _supertonic package under a unique name and return its
    ``convert`` submodule (load_models / convert_all)."""
    import importlib

    pkg_name = "supertonic3_xnnpack_supertonic"
    pkg_init = _MODEL_DIR / "_supertonic" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        pkg_name, pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_supertonic")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module(f"{pkg_name}.convert")


@contextlib.contextmanager
def _size_oblivious_export():
    """Treat backed sizes size-obliviously so the VITS relative-attention
    reshapes export with a dynamic sequence length (their divisibility
    identities are always true but torch.export cannot prove them). Must stay
    active through BOTH export and lowering, otherwise the memory planner
    reserves a phantom >512 MB buffer for the text encoder's quadratic
    attention that overflows the runtime arena at load time."""
    import torch.fx.experimental._config as fx_config

    prev = fx_config.backed_size_oblivious
    fx_config.backed_size_oblivious = True
    try:
        yield
    finally:
        fx_config.backed_size_oblivious = prev


def _download_onnx_dir() -> str:
    from huggingface_hub import hf_hub_download

    local = None
    for rel in _ONNX_FILES:
        p = hf_hub_download(_ONNX_REPO, rel)
        local = Path(p).parent
    return str(local)


def _example_inputs(t_text=64, t_latent=80):
    return dict(
        text_ids=torch.randint(1, 8000, (1, t_text), dtype=torch.long),
        text_mask=torch.ones(1, 1, t_text),
        style_ttl=torch.randn(1, _N_STYLE, _STYLE_DIM),
        style_dp=torch.randn(1, *_DP_STYLE),
        noisy_latent=torch.randn(1, _LATENT_DIM, t_latent),
        text_emb=torch.randn(1, _TEXT_EMB_DIM, t_text),
        latent_mask=torch.ones(1, 1, t_latent),
        current_step=torch.tensor([2.0]),
        total_step=torch.tensor([8.0]),
    )


def _method_spec(models, ex):
    """(model, example inputs tuple, dynamic_shapes tuple) per sub-model name.
    Each method gets its own fresh Dim objects."""
    from torch.export import Dim

    return {
        "duration_predictor": (
            models["duration_predictor"],
            (ex["text_ids"], ex["style_dp"], ex["text_mask"]),
            ({1: Dim("T", min=_T_MIN, max=_T_MAX)}, None, {2: Dim("T", min=_T_MIN, max=_T_MAX)}),
        ),
        "text_encoder": (
            models["text_encoder"],
            (ex["text_ids"], ex["style_ttl"], ex["text_mask"]),
            ({1: Dim("T", min=_T_MIN, max=_T_MAX)}, None, {2: Dim("T", min=_T_MIN, max=_T_MAX)}),
        ),
        "vector_estimator": (
            models["vector_estimator"],
            (ex["noisy_latent"], ex["text_emb"], ex["style_ttl"], ex["text_mask"],
             ex["latent_mask"], ex["current_step"], ex["total_step"]),
            ({2: Dim("L", min=_L_MIN, max=_L_MAX)}, {2: Dim("T", min=_T_MIN, max=_T_MAX)}, None,
             {2: Dim("T", min=_T_MIN, max=_T_MAX)}, {2: Dim("L", min=_L_MIN, max=_L_MAX)}, None, None),
        ),
        "vocoder": (
            models["vocoder"],
            (ex["noisy_latent"],),
            ({2: Dim("L", min=_L_MIN, max=_L_MAX)},),
        ),
    }


def _submodel_for(file_name: str) -> str:
    for name in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder"):
        if file_name.startswith(name):
            return name
    raise ValueError(f"unknown supertonic sub-model for variant file '{file_name}'")


def export(config: dict, out_dir: str):
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    convert = _load_convert()
    onnx_dir = _download_onnx_dir()
    models = convert.load_models(onnx_dir=onnx_dir)

    all_schemas = json.loads(load_and_validate_schema(_MODEL_DIR / "schema.json"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Size-oblivious mode wraps both export and lowering (see _size_oblivious_export).
    with _size_oblivious_export(), torch.inference_mode():
        ex = _example_inputs()
        specs = _method_spec(models, ex)

        for variant in config["variants"]:
            name = _submodel_for(variant["file"])
            model, inputs, dynamic_shapes = specs[name]

            ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes, strict=False)

            sub_schema_json = json.dumps({"forward": all_schemas[name]})

            program = to_edge_transform_and_lower(
                {"forward": ep},
                partitioner=[XnnpackPartitioner()],
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
                ),
                transform_passes=[AddmmToLinearTransform()],
                constant_methods={"get_model_schema": sub_schema_json},
            ).to_executorch()

            out_path = out_dir / variant["file"]
            out_path.write_bytes(program.buffer)
            written.append(str(out_path))

    return written
