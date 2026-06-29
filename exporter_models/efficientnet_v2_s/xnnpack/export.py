from pathlib import Path

import torch
import torchvision.models as tvm
from torch.export import export as torch_export

from exporter.backends import lower


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    model = tvm.efficientnet_v2_s(
        weights=tvm.EfficientNet_V2_S_Weights.DEFAULT
    ).eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
