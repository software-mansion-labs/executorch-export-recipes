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

# Per-arch int8 recipe overrides. MobileNetV3 loses 1.72 mIoU fully int8 (69.13
# fp32 -> 67.41 on 300 VOC2012 val images); its stem and final 1x1 classifier
# conv carry most of that, and keeping both fp32 with 500 calibration images
# brings it to 68.88. The ResNet backbones are within 0.4 of fp32 as they are.
_FLOAT_MODULES = {
    "deeplab_v3_mobilenet_v3_large": ["model.backbone.0", "model.classifier.4"],
}
_CALIBRATION_SIZES = {"deeplab_v3_mobilenet_v3_large": 500}

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


class _Float(torch.nn.Module):
    """Marks a submodule the quantizer must leave in fp32."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


class _DeeplabWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The shipped .pte returns only the main "out" head (no aux).
        return self.model(x)["out"]


def _arch_for(file_name: str) -> str:
    # longest match first so resnet101 isn't shadowed by resnet50, etc.
    for arch in sorted(_ARCHS, key=len, reverse=True):
        if file_name.startswith(arch):
            return arch
    raise ValueError(f"unknown deeplab arch for variant file '{file_name}'")


def _calibrate(prepared, count):
    import torchvision.transforms.functional as F
    from torchvision import datasets
    from torchvision.io import decode_image

    random.seed(0)
    ds = datasets.VOCSegmentation(
        root="datasets/VOCSegmentation",
        year="2012",
        image_set="train",
        # download=True re-fetches the 2 GB tarball even when it is already
        # extracted, and the host throttles; only download when it is missing.
        download=not Path("datasets/VOCSegmentation/VOCdevkit/VOC2012").exists(),
    )
    for img_path in random.sample(ds.images, k=count):
        img = decode_image(img_path).unsqueeze(0)
        img = F.resize(img, _RESIZE, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        prepared(img)


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    arch = _arch_for(variant["file"])
    builder, weights = _ARCHS[arch]
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
        # set_module_name(name, None) is rejected by XNNPACKQuantizer, while
        # set_module_type(type, None) works, hence the _Float marker.
        for name in _FLOAT_MODULES.get(arch, []):
            parent, _, child = name.rpartition(".")
            owner = model.get_submodule(parent)
            setattr(owner, child, _Float(getattr(owner, child)))
        quantizer.set_module_type(_Float, None)
        prepared = prepare_pt2e(torch_export(model, inputs).module(), quantizer)
        _calibrate(prepared, _CALIBRATION_SIZES.get(arch, _CALIBRATION_SIZE))
        model = convert_pt2e(prepared)

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
