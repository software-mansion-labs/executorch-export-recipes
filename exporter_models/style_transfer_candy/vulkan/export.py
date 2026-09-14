import re
import tempfile
import zipfile
from pathlib import Path

import torch
from torch.export import export as torch_export
from torch.hub import download_url_to_file


# -----------------------------------------------------------------------------
# TransformerNet — vendored verbatim from exports/style-transfer/transformer_net.py
# Source: https://github.com/pytorch/examples/tree/main/fast_neural_style
# BSD 3-Clause License — Copyright (c) 2017, Pytorch contributors
# -----------------------------------------------------------------------------


def _reflect_pad(x, pad):
    left   = x[:, :, :, 1:pad + 1].flip(-1)
    right  = x[:, :, :, -(pad + 1):-1].flip(-1)
    x = torch.cat([left, x, right], dim=-1)
    top    = x[:, :, 1:pad + 1, :].flip(-2)
    bottom = x[:, :, -(pad + 1):-1, :].flip(-2)
    return torch.cat([top, x, bottom], dim=-2)


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        y = y.clamp(0, 255) / 255.0
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.pad = kernel_size // 2
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv2d(_reflect_pad(x, self.pad))


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.pad = kernel_size // 2
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = torch.nn.functional.interpolate(x, mode="bilinear", scale_factor=self.upsample)
        return self.conv2d(_reflect_pad(x, self.pad))


# -----------------------------------------------------------------------------

_WEIGHTS_ZIP_URL = "https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1"


def _load_candy_state_dict() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "saved_models.zip"
        download_url_to_file(_WEIGHTS_ZIP_URL, str(zip_path), progress=False)
        with zipfile.ZipFile(zip_path) as zf:
            candy_name = next(n for n in zf.namelist() if n.endswith("candy.pth"))
            zf.extract(candy_name, tmp)
        state_dict = torch.load(Path(tmp) / candy_name, map_location="cpu", weights_only=True)
    for k in list(state_dict.keys()):
        if re.search(r"in\d+\.running_(mean|var)$", k):
            del state_dict[k]
    return state_dict


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    model = TransformerNet()
    model.load_state_dict(_load_candy_state_dict())
    model.eval()

    inputs = (torch.randn(shape, dtype=torch.float32),)

    ep = torch_export(model, inputs)
    program = to_edge_transform_and_lower(
        ep,
        partitioner=[
            VulkanPartitioner(
                compile_options={"force_fp16": True}
                if variant["precision"] == "fp16"
                else {}
            )
        ],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
