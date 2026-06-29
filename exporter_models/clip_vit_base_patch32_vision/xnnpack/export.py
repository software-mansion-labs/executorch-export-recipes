from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import export as torch_export

from exporter.backends import lower

_SOURCE = "openai/clip-vit-base-patch32"


class _ClipVisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModel.from_pretrained(_SOURCE)
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values):
        pooled = self.vision_model(pixel_values).pooler_output
        projected = self.visual_projection(pooled)
        return F.normalize(projected, p=2, dim=1)


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    module = _ClipVisionEncoder().eval()
    inputs = (torch.zeros(shape, dtype=torch.float32),)

    ep = torch_export(module, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
