import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# 4-bit weight-only, LINEARS ONLY. Quantizing the convolutions destroys SDXS
# (see NOTES.md), so the convs and the embeddings stay fp32.
QLINEAR_CONFIG = "4w"
QLINEAR_GROUP_SIZE = 32


def _load_pipeline():
    path = _MODEL_DIR / "_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sdxs_512_dreamshaper_mlx_pipeline", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sdxs_512_dreamshaper_mlx_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch import exir
    from executorch.backends.mlx import MLXPartitioner

    # MLX-native quantization (NOT torchao / extension.llm): the runtime MLX
    # backend only executes weights packed by this module.
    from executorch.backends.mlx.llm.quantization import quantize_model_
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig

    pipeline = _load_pipeline()

    variant = config["variant"]
    methods = pipeline.load_methods()

    if variant["quantized"]:
        for module in methods.values():
            quantize_model_(
                module,
                qlinear_config=QLINEAR_CONFIG,
                qlinear_group_size=QLINEAR_GROUP_SIZE,
                skip_incompatible_shapes=True,
            )
            module.eval()

    inputs = pipeline.example_inputs(variant["methods"])
    programs = pipeline.exported_programs(methods, inputs)

    executorch_program = exir.to_edge_transform_and_lower(
        programs,
        transform_passes=get_default_passes(),
        partitioner={name: [MLXPartitioner()] for name in programs},
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(executorch_program.buffer)
    return str(out_path)
