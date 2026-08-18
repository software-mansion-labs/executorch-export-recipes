import importlib.util
import sys
import random
from pathlib import Path

import torch
from torch.export import export as torch_export

from exporter.backends import lower

_CALIBRATION_SIZE = 100

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_STYLE = "rain_princess"


def _load_net():
    name = "style_rain_princess_xnnpack_net"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_transformer_net.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
        # TransformerNet takes 0-255 floats (upstream fast_neural_style mul(255)).
        prepared(img.float())


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    model = _load_net().load_style(_STYLE)
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
