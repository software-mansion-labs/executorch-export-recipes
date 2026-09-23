from pathlib import Path

import torch
import torchvision.models.segmentation as tvs
from torch.export import export as torch_export


_RESIZE = [520, 520]
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_ARCHS = {
    "fcn_resnet50": (tvs.fcn_resnet50, lambda: tvs.FCN_ResNet50_Weights.DEFAULT),
    "fcn_resnet101": (tvs.fcn_resnet101, lambda: tvs.FCN_ResNet101_Weights.DEFAULT),
}


class _FcnWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The shipped .pte returns only the main "out" head (no aux).
        return self.model(x)["out"]


def _arch_for(file_name: str):
    for arch in sorted(_ARCHS, key=len, reverse=True):
        if file_name.startswith(arch):
            return _ARCHS[arch]
    raise ValueError(f"unknown fcn arch for variant file '{file_name}'")


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    builder, weights = _arch_for(variant["file"])
    inputs = (torch.randn(shape, dtype=torch.float32),)

    model = _FcnWrapper(builder(weights=weights())).eval()

    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

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
