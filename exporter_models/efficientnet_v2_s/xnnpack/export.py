import random
from pathlib import Path

import torch
import torchvision.models as tvm
from torch.export import export as torch_export

from exporter.backends import lower

_CALIBRATION_SIZE = 500

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
    from torchvision.io import ImageReadMode, decode_image

    # ImageNet-domain calibration (Imagenette is a 10-class ImageNet subset):
    # 73.1% top-1 on ImageNetV2 against 72.7% for the VOC images used before.
    random.seed(0)
    root = Path("datasets/Imagenette")
    ds = datasets.Imagenette(root=str(root), split="train", size="320px", download=not root.exists())
    for img_path, _ in random.sample(ds._samples, k=_CALIBRATION_SIZE):
        img = decode_image(str(img_path), mode=ImageReadMode.RGB).unsqueeze(0)
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
        # The squeeze-excitation gates are what int8 breaks: keeping them fp32
        # takes ImageNetV2 top-1 from 72.7% to 75.4% (fp32 75.8%).
        from torchvision.ops.misc import SqueezeExcitation

        quantizer.set_module_type(SqueezeExcitation, None)
        prepared = prepare_pt2e(torch_export(model, inputs).module(), quantizer)
        _calibrate(prepared, shape)
        model = convert_pt2e(prepared)

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
