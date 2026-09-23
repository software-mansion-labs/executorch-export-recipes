import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export


_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_wrapper():
    pkg_init = _MODEL_DIR / "_wrappers" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "yolo26_seg_vulkan_wrappers", pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_wrappers")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["yolo26_seg_vulkan_wrappers"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from ultralytics import YOLO

    YOLOInstanceSegmentation = _load_wrapper().YOLOInstanceSegmentation

    size = config.get("size") or "n"
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])  # [1,3,R,R]
    resolution = shape[-1]

    yolo = YOLO(f"yolo26{size}-seg.pt")
    model = YOLOInstanceSegmentation(yolo.model, method_size=resolution).eval()
    inputs = (torch.randn(*shape, dtype=torch.float32),)
    with torch.no_grad():
        model(*inputs)  # warmup: fix export-mode head shapes

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
