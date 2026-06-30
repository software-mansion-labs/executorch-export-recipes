import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import export as torch_export

from exporter.backends import lower

_SOURCE = "openai/clip-vit-base-patch32"
_CALIBRATION_SIZE = 1000


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


def _calibrate(prepared):
    from transformers import AutoProcessor
    from torchvision import datasets
    from torchvision.io import decode_image

    processor = AutoProcessor.from_pretrained(_SOURCE)
    random.seed(0)
    ds = datasets.VOCSegmentation(
        root="datasets/VOCSegmentation", year="2012", image_set="train", download=True
    )
    for img_path in random.sample(ds.images, k=_CALIBRATION_SIZE):
        img = decode_image(img_path)
        pixel_values = processor(images=img, return_tensors="pt")["pixel_values"]
        prepared(pixel_values)


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    module = _ClipVisionEncoder().eval()
    inputs = (torch.zeros(shape, dtype=torch.float32),)

    if variant["quantized"]:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(torch_export(module, inputs).module(), quantizer)
        _calibrate(prepared)
        module = convert_pt2e(prepared)

    ep = torch_export(module, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
