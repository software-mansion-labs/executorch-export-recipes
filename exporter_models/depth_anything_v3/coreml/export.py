import importlib.util
import sys
from pathlib import Path

import torch


def _load_model():
    name = "depth_anything_v3_model_coreml"
    path = Path(__file__).resolve().parents[1] / "_model.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_model()


def _static_rope(self, tokens, positions):
    """Use the known patch-grid extent instead of extracting a tensor scalar."""
    feature_dim = tokens.size(-1) // 2
    cos, sin = self._compute_frequency_components(
        feature_dim, self._export_max_position, tokens.device, tokens.dtype
    )
    vertical, horizontal = tokens.chunk(2, dim=-1)
    return torch.cat(
        (
            self._apply_1d_rope(vertical, positions[..., 0], cos, sin),
            self._apply_1d_rope(horizontal, positions[..., 1], cos, sin),
        ),
        dim=-1,
    )


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    model = _load_model()
    from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D

    for layer in model.modules():
        if isinstance(layer, RotaryPositionEmbedding2D):
            layer._export_max_position = max(shape[-2:]) // 14 + 1
            layer.forward = _static_rope.__get__(layer)
    torch.set_num_threads(4)
    inputs = (torch.zeros(shape, dtype=torch.float32),)
    with torch.no_grad():
        model(*inputs)
        ep = torch.export.export(model, inputs, strict=False)
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_unit=ct.ComputeUnit.ALL,
    )
    program = to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
    ).to_executorch()

    path = Path(out_dir) / variant["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(program.buffer)
    return str(path)
