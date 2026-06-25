from pathlib import Path

import torch
import torchvision.models.detection as tvd
from torch.export import export as torch_export

from exporter.backends import lower


class _SsdliteWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = tvd.ssdlite320_mobilenet_v3_large(
            weights=tvd.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )

    def forward(self, image):
        det = self.model([image])[0]
        return det["boxes"], det["scores"], det["labels"]


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    model = _SsdliteWrapper().eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
