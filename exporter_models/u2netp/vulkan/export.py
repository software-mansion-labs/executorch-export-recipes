import copy
import importlib.util
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent.parent


def _load_wrapper():
    key = "u2netp_wrapper"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / "_wrapper.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


class _Float32Convolution(torch.nn.Module):
    def __init__(self, convolution):
        super().__init__()
        self.convolution = convolution

    def forward(self, image):
        return self.convolution(image.float()).half()


class _Float16Network(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image.half()).float()


def _prepare_fp16(model):
    # Fuse before casting: unfused BatchNorm parameters are unsafe in FP16.
    for block in model.model.modules():
        if hasattr(block, "conv_s1") and hasattr(block, "bn_s1"):
            block.conv_s1 = torch.nn.utils.fusion.fuse_conv_bn_eval(block.conv_s1, block.bn_s1)
            block.bn_s1 = torch.nn.Identity()
    # This convolution is unusually sensitive to reduced precision. Retain its
    # original FP32 weights, compute and bias; cast its result back to FP16.
    sensitive = copy.deepcopy(model.model.stage2d.rebnconv1d.conv_s1)
    model.model.half()
    model.model.stage2d.rebnconv1d.conv_s1 = _Float32Convolution(sensitive)
    model.model = _Float16Network(model.model)
    return model


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    wrapper = _load_wrapper()
    model = wrapper.U2NetPForRNE(wrapper.load_model()).eval()
    model = _prepare_fp16(model)
    inputs = (torch.rand(*shape, dtype=torch.float32),)
    ep = torch.export.export(model, inputs, strict=True)
    program = to_edge_transform_and_lower(
        ep,
        partitioner=[VulkanPartitioner(compile_options={
            "texture_limits": (2048, 2048, 2048),
        })],
    ).to_executorch()
    path = Path(out_dir) / variant["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(program.buffer)
    return str(path)
