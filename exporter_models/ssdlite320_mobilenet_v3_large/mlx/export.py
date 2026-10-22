import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_model_module():
    pkg_init = _MODEL_DIR / "_model" / "__init__.py"
    module_name = f"_vendored_{_MODEL_DIR.name}_mlx_model"
    spec = importlib.util.spec_from_file_location(
        module_name, pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_model")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class _SsdliteWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        det = self.model([image])[0]
        return det["boxes"], det["scores"], det["labels"].float()


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    m = _load_model_module()
    base_model = m.ssdlite320_mobilenet_v3_large(
        weights=m.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    ).eval()
    model = _SsdliteWrapper(base_model).eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    ep = torch_export(model, inputs)
    program = to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
