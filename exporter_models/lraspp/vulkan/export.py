from pathlib import Path

import torch


def _vk_argmax(logits, k: int):
    """argmax(dim=1) built only from ops the Vulkan partitioner takes.

    Ported from the Core ML index-map exports. Vulkan registers amax and
    eq.Tensor, but `is_reduce_node_supported` rejects keepdim=False on any dim
    but the last, so both reductions keep their dim and the result is squeezed
    afterwards.

    torch.argmax returns the FIRST maximal index while (mask * arange).amax
    returns the last, hence the reversal.
    """
    ar = torch.arange(k, dtype=logits.dtype, device=logits.device).view(1, k, 1, 1)
    peak = logits.amax(dim=1, keepdim=True)
    hit = (logits == peak).to(logits.dtype)
    idx = (((k - 1) - ar) * hit).amax(dim=1, keepdim=True)
    return ((k - 1) - idx).squeeze(1)


class _LrasppWrapper(torch.nn.Module):
    def __init__(self, model, index_map: bool):
        super().__init__()
        self.model = model
        self.index_map = index_map

    def forward(self, x):
        logits = self.model(x)["out"]
        if not self.index_map:
            return logits
        # 1x21x520x520 fp32 is 22.7 MB off the GPU every inference; the index
        # map is 1.08 MB and the task argmaxes these logits away immediately.
        return _vk_argmax(logits, 21).to(torch.int32)


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

    index_map = "index" in variant["file"]
    model = _LrasppWrapper(
        tvs.lraspp_mobilenet_v3_large(
            weights=tvs.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        ),
        index_map,
    ).eval()

    ep = torch_export(model, inputs)
    # fp16 is the point of running this on the GPU; fp32 Vulkan measured about
    # 2x slower than XNNPACK on both Adreno and Mali for the sibling
    # selfie_segmentation export.
    program = to_edge_transform_and_lower(
        ep,
        partitioner=[VulkanPartitioner(compile_options={"force_fp16": True})],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
