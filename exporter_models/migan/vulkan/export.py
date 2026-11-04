import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_migan():
    key = "migan_vulkan_vendored"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / "_migan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod



def export(config: dict, out_dir: str) -> str:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from torch.export import export as torch_export

    mg = _load_migan()
    size = config["size"]
    variant = config["variant"]
    inputs_spec = variant["methods"]["forward"]["inputs"]
    image_shape = tuple(inputs_spec[0]["shape"])
    mask_shape = tuple(inputs_spec[1]["shape"])
    resolution = image_shape[-1]
    assert image_shape[-2:] == mask_shape[-2:] == (mg.RESOLUTION_FOR_SIZE[size],) * 2, (
        f"config shapes {image_shape}/{mask_shape} disagree with size {size} ({mg.RESOLUTION_FOR_SIZE[size]})"
    )

    generator = mg.Generator(resolution=resolution)
    weights = mg.fetch_weights(mg.CHECKPOINT_FOR_SIZE[size])
    generator.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    generator = mg.vulkanize(generator)  # see Downsample2dVulkan in _migan.py
    generator = mg.fold_activation_gain(generator)  # see LreluClampFolded in _migan.py
    model = mg.InpaintingWrapper(generator)
    model.eval()

    inputs = (
        torch.rand(*image_shape, dtype=torch.float32),
        (torch.rand(*mask_shape) > 0.3).to(torch.float32),
    )
    ep = torch_export(model, inputs, strict=True)
    # fp16 compute on the GPU (force_fp16); the graph itself stays fp32 with
    # fp32 I/O, so the app-side contract is the same as xnnpack / coreml.
    compile_options = {"force_fp16": True} if variant["precision"] == "fp16" else {}
    program = to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner(compile_options=compile_options)]
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
