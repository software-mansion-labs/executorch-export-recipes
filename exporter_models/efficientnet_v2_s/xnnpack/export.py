import random
from pathlib import Path

import torch
import torchvision.models as tvm
from torch.export import export as torch_export

from exporter.backends import lower

_CALIBRATION_SIZE = 100

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class _EfficientNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # The shipped .pte normalizes the raw [0,1] image inside the graph.
        return self.model((x - self.mean) / self.std)


def _calibrate(prepared, shape):
    import torchvision.transforms.functional as F
    from torchvision import datasets
    from torchvision.io import decode_image

    random.seed(0)
    ds = datasets.VOCSegmentation(
        root="datasets/VOCSegmentation", year="2012", image_set="train", download=True
    )
    for img_path in random.sample(ds.images, k=_CALIBRATION_SIZE):
        img = decode_image(img_path).unsqueeze(0)
        img = F.resize(
            img, list(shape[-2:]), interpolation=F.InterpolationMode.BILINEAR, antialias=True
        )
        # _EfficientNetWrapper normalizes inside the graph, so feed raw [0,1].
        prepared(F.convert_image_dtype(img, torch.float))


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    model = _EfficientNetWrapper(
        tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.DEFAULT)
    ).eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    if variant["quantized"]:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(torch_export(model, inputs).module(), quantizer)
        _calibrate(prepared, shape)
        model = convert_pt2e(prepared)

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
