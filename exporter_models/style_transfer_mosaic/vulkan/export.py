import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export


_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_STYLE = "mosaic"


def _load_net():
    name = "style_mosaic_vulkan_net"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_transformer_net.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    model = _load_net().load_style(_STYLE)
    inputs = (torch.randn(shape, dtype=torch.float32),)

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
