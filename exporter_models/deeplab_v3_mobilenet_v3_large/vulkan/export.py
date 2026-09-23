from pathlib import Path

import torch


class _DeeplabWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The shipped .pte returns only the main "out" head (no aux).
        return self.model(x)["out"]


def export(config: dict, out_dir: str) -> str:
    import torchvision.models.segmentation as tvs
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower
    from torch.export import export as torch_export

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    inputs = (torch.randn(shape, dtype=torch.float32),)

    # Only the MobileNetV3 backbone is a Vulkan candidate; the ResNet variants
    # measured numerically wrong on Vulkan, see the campaign notes.
    model = _DeeplabWrapper(
        tvs.deeplabv3_mobilenet_v3_large(
            weights=tvs.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
    ).eval()

    ep = torch_export(model, inputs)
    program = to_edge_transform_and_lower(
        ep,
        partitioner=[VulkanPartitioner(compile_options={"force_fp16": True})],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
