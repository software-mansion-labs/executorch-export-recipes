import importlib.util
import sys
from pathlib import Path

import torch


def _load_model():
    name = "zipdepth_model_vulkan"
    path = Path(__file__).resolve().parents[1] / "_model.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    model = module.build_model()
    _expand_grouped_convolutions(model)
    from zipdepth.model.architecture import MinimalCrossScale

    for layer in model.modules():
        if isinstance(layer, MinimalCrossScale):
            layer.forward = _cross_scale_regular_pool.__get__(layer)
    return model


def _cross_scale_regular_pool(self, x_high, x_low):
    """Use equivalent fixed pooling when the two feature grids divide exactly."""
    height, width = x_high.shape[-2:]
    low_height, low_width = x_low.shape[-2:]
    assert height % low_height == 0 and width % low_width == 0
    kernel = (height // low_height, width // low_width)
    low_up = torch.nn.functional.interpolate(
        self.low_to_high(x_low), size=(height, width), mode="nearest"
    )
    high_down = torch.nn.functional.avg_pool2d(
        self.high_to_low(x_high), kernel_size=kernel, stride=kernel
    )
    return x_high + low_up * 0.3, x_low + high_down * 0.3


def _expand_grouped_convolutions(module):
    """Use equivalent block-diagonal dense weights for grouped Vulkan convs.

    ExecuTorch 1.4.1 accepts grouped non-depthwise convolutions during
    partitioning, but its Vulkan pointwise shader reads the first input group
    for every output group. Dense weights avoid that shader limitation.
    """
    for name, child in list(module.named_children()):
        if (
            isinstance(child, torch.nn.Conv2d)
            and child.groups > 1
            and not (child.groups == child.in_channels == child.out_channels)
        ):
            dense = torch.nn.Conv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=1,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            ).eval()
            inputs = child.in_channels // child.groups
            outputs = child.out_channels // child.groups
            with torch.no_grad():
                dense.weight.zero_()
                for group in range(child.groups):
                    dense.weight[
                        group * outputs : (group + 1) * outputs,
                        group * inputs : (group + 1) * inputs,
                    ].copy_(child.weight[group * outputs : (group + 1) * outputs])
                if child.bias is not None:
                    dense.bias.copy_(child.bias)
            setattr(module, name, dense)
        else:
            _expand_grouped_convolutions(child)


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    torch.set_num_threads(4)
    model = _load_model()
    inputs = (torch.zeros(shape, dtype=torch.float32),)
    with torch.no_grad():
        ep = torch.export.export(model, inputs, strict=False)
    program = to_edge_transform_and_lower(
        ep,
        partitioner=[
            VulkanPartitioner(
                compile_options={
                    "force_fp16": True,
                    "texture_limits": (2048, 2048, 2048),
                }
            )
        ],
    ).to_executorch()

    path = Path(out_dir) / variant["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(program.buffer)
    return str(path)
