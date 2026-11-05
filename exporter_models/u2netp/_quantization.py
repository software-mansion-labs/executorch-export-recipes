"""Selective static INT8 for the convolutions that dominate XNNPACK latency."""

import urllib.request
from pathlib import Path

import torch

# Keep sensitive convolutions, residuals and concatenations in FP32.
_INT8_MODULES = {
    "model.stage1d.rebnconvin",
    "model.stage1d.rebnconv1d",
    "model.stage2d.rebnconvin",
    "model.stage2.rebnconvin",
}
_UPSTREAM = "https://raw.githubusercontent.com/xuebinqin/U-2-Net/ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29/test_data/test_images"
_CALIBRATION_IMAGES = (
    "0002-01.jpg", "0003.jpg", "girl.png", "hockey.png", "horse.jpg",
    "im_01.png", "im_14.png", "im_21.png", "im_27.png",
    "lamp2_meitu_1.jpg", "long.jpg", "rifle1.jpg", "sailboat3.jpg",
)


def quantize(model: torch.nn.Module, inputs: tuple) -> torch.fx.GraphModule:
    from dataclasses import replace

    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from PIL import Image
    from torchao.quantization.pt2e.observer import PerChannelMinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    config = get_symmetric_quantization_config(is_per_channel=True)
    config = replace(config, weight=replace(
        config.weight,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(
            eps=torch.finfo(torch.float32).eps,
        ),
    ))
    # ET 1.4.1's cat annotator ignores its filter. Restrict operator selection
    # to convolutions so excluded skip connections really remain float32.
    quantizer = XNNPACKQuantizer().set_operator_type(torch.ops.aten.conv2d.default, config)
    quantizer.set_filter_function(
        lambda node: node.op == "get_attr" or any(
            path in _INT8_MODULES
            for path, _ in node.meta.get("nn_module_stack", {}).values()
        )
    )
    prepared = prepare_pt2e(torch.export.export(model, inputs, strict=True).module(), quantizer)
    cache = Path(__file__).resolve().parents[2] / "datasets" / "u2netp" / "calibration"
    cache.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for name in _CALIBRATION_IMAGES:
            path = cache / name
            if not path.exists():
                with urllib.request.urlopen(f"{_UPSTREAM}/{name}", timeout=60) as response:
                    data = response.read()
                path.write_bytes(data)
            with Image.open(path) as image:
                image = image.convert("RGB").resize((320, 320), Image.Resampling.BILINEAR)
                tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
                tensor = tensor.reshape(320, 320, 3).permute(2, 0, 1).contiguous()
                prepared(tensor.unsqueeze(0).float() / 255)
        return convert_pt2e(prepared)
