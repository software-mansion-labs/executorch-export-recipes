import importlib.util
import sys
from pathlib import Path

import torch


def _load_model():
    name = "depth_anything_v2_model_vulkan"
    path = Path(__file__).resolve().parents[1] / "_model.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_model()


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    model = _load_model()
    inputs = (torch.zeros(shape, dtype=torch.float32),)
    with torch.no_grad():
        ep = torch.export.export(model, inputs, strict=False)
    program = to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner(compile_options={
            "force_fp16": True, "texture_limits": (2048, 2048, 2048),
        })],
    ).to_executorch()

    path = Path(out_dir) / variant["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(program.buffer)
    return str(path)
