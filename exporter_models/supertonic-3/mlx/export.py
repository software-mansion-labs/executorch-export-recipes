"""Supertonic 3 -> ExecuTorch MLX (multi_pte).

Exports the four Supertonic 3 sub-models lowered to the MLX backend (Apple Silicon GPU):

    duration_predictor (text_ids, style_dp,  text_mask)               -> duration
    text_encoder      (text_ids, style_ttl, text_mask)               -> text_emb
    vector_estimator  (noisy_latent, text_emb, style_ttl, text_mask,
                       latent_mask, current_step, total_step)         -> denoised_latent
    vocoder           (latent)                                        -> wav

Each .pte exposes two methods:
  - forward          — sub-model inference (MLX-delegated)
  - get_model_schema — companion method returning the sub-model spec JSON {"forward": ...}

NOTE (CUSTOM BUILD REQUIREMENT):
  Running MLX models in react-native-executorch on device requires a custom build
  of ExecuTorch from Software Mansion (https://github.com/software-mansion-labs/executorch).
  Specifically:
    1. SWM's custom MLX pattern fusions (select-as-view, RMSNorm decomposition)
       produce the 2-subgraph structure expected by the C++ MLX interpreter.
    2. SWM's C++ MLX runtime frees intermediate temp tensors at last use during
       the 8-step flow-matching loop, preventing GPU memory accumulation (OOM/Error::Internal).
"""

import contextlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

_ONNX_REPO = "Supertone/supertonic-3"
_ONNX_FILES = [
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
]

_T_MIN, _T_MAX = 8, 256
_L_MIN, _L_MAX = 4, 1000

_N_STYLE = 50
_STYLE_DIM = 256
_DP_STYLE = (8, 16)
_TEXT_EMB_DIM = 256
_LATENT_DIM = 144


def _ensure_swm_mlx_backend():
    """Ensure Software Mansion's MLX exporter backend is active in Python.

    If `uv sync` restored stock PyPI ExecuTorch, this function automatically syncs
    Software Mansion's custom MLX backend (including pattern passes and FlatBuffer
    schema serializer) into site-packages.
    """
    if importlib.util.find_spec("executorch.backends.mlx._generated_inspector") is not None:
        return

    print("[0/3] Software Mansion MLX backend not detected in venv. Syncing...")
    cache_dir = _MODEL_DIR / "scratch_swm" / "executorch"

    if not cache_dir.exists():
        print("  - cloning software-mansion-labs/executorch...")
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/software-mansion-labs/executorch.git", str(cache_dir)],
            check=True,
        )

    print("  - generating FlatBuffer schema...")
    ser_dir = cache_dir / "backends" / "mlx" / "serialization"
    subprocess.run([sys.executable, "generate.py"], cwd=ser_dir, check=True)

    print("  - installing MLX backend into site-packages...")
    import executorch.backends.mlx as mlx_pkg
    target_dir = Path(mlx_pkg.__file__).parent
    shutil.rmtree(target_dir)
    shutil.copytree(cache_dir / "backends" / "mlx", target_dir)
    print("  - SWM MLX backend synced successfully!")


def _load_convert():
    import importlib

    pkg_name = "supertonic3_mlx_supertonic"
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
    _ensure_swm_mlx_backend()

    from executorch.backends.mlx import MLXPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    convert = _load_convert()
    onnx_dir = _download_onnx_dir()
    models = convert.load_models(onnx_dir=onnx_dir)

    all_schemas = json.loads(load_and_validate_schema(_MODEL_DIR / "schema.json"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

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
                partitioner=[MLXPartitioner()],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
                constant_methods={"get_model_schema": sub_schema_json},
            ).to_executorch()

            out_path = out_dir / variant["file"]
            out_path.write_bytes(program.buffer)
            written.append(str(out_path))

    return written
