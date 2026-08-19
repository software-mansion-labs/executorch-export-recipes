import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_model():
    name = "rfdetr_kp_model_mlx"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_keypoint_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    # Grid-free deformable-attention core + constant-folded positional embedding:
    # both replace ops the MLX delegate has no handler for (grid_sampler_2d,
    # pow.Scalar), which would otherwise fragment the graph into CPU subgraphs.
    wrapper, example = _load_model().build_wrapper(
        patch_flags={"ms_deform_attn_grid_free": True, "pos_embed_compile_safe": True}
    )
    with torch.inference_mode():
        ep = torch_export(wrapper, args=example, strict=False)

    program = to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        # MLXReinplacePass rewrites elementwise chains into aten in-place ops
        # (add_/mul_/...), which are not core ATen; every other MLX export here
        # uses the same config for that reason.
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
