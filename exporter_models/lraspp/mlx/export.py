from pathlib import Path

import torch
import torchvision.models.segmentation as tvs
from torch.export import export as torch_export


class _SegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The shipped .pte returns only the main "out" head (no aux).
        return self.model(x)["out"]


def _build(file_name: str):
    return tvs.lraspp_mobilenet_v3_large(
        weights=tvs.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    )


def _lower(ep, precision: str):
    from executorch import exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig

    return exir.to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    ).to_executorch()


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    inputs = (torch.randn(shape, dtype=torch.float32),)

    model = _SegWrapper(_build(variant["file"])).eval()
    ep = torch_export(model, inputs)
    program = _lower(ep, variant["precision"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
