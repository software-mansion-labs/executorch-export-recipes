import random
from pathlib import Path

import torch
import torchvision.models.segmentation as tvs
from torch.export import export as torch_export

from exporter.backends import lower

_RESIZE = [520, 520]
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_CALIBRATION_SIZE = 100

_ARCHS = {
    "deeplab_v3_mobilenet_v3_large": (
        tvs.deeplabv3_mobilenet_v3_large,
        lambda: tvs.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
    ),
    "deeplab_v3_resnet50": (
        tvs.deeplabv3_resnet50,
        lambda: tvs.DeepLabV3_ResNet50_Weights.DEFAULT,
    ),
    "deeplab_v3_resnet101": (
        tvs.deeplabv3_resnet101,
        lambda: tvs.DeepLabV3_ResNet101_Weights.DEFAULT,
    ),
}


class _DeeplabWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["out"], out["aux"]


def _arch_for(file_name: str):
    # longest match first so resnet101 isn't shadowed by resnet50, etc.
    for arch in sorted(_ARCHS, key=len, reverse=True):
        if file_name.startswith(arch):
            return _ARCHS[arch]
    raise ValueError(f"unknown deeplab arch for variant file '{file_name}'")


def _calibrate(prepared):
    import torchvision.transforms.functional as F
    from torchvision import datasets
    from torchvision.io import decode_image

    random.seed(0)
    ds = datasets.VOCSegmentation(
        root="datasets/VOCSegmentation", year="2012", image_set="train", download=True
    )
    for img_path in random.sample(ds.images, k=_CALIBRATION_SIZE):
        img = decode_image(img_path).unsqueeze(0)
        img = F.resize(img, _RESIZE, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        prepared(img)


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    builder, weights = _arch_for(variant["file"])
    inputs = (torch.randn(shape, dtype=torch.float32),)

    model = _DeeplabWrapper(builder(weights=weights())).eval()

    if variant["quantized"]:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(torch_export(model, inputs).module(), quantizer)
        _calibrate(prepared)
        model = convert_pt2e(prepared)

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
