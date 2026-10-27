import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_migan():
    key = "migan_coreml_vendored"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / "_migan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod



def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
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
    generator = mg.fold_activation_gain(generator)  # see LreluClampFolded in _migan.py
    generator = mg.upsample_as_deconv(generator)  # see Upsample2dDeconv in _migan.py
    model = mg.InpaintingWrapper(generator)
    model.eval()

    inputs = (
        torch.rand(*image_shape, dtype=torch.float32),
        (torch.rand(*mask_shape) > 0.3).to(torch.float32),
    )
    ep = torch_export(model, inputs, strict=True)
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[variant["precision"]]
    # Swept on an iPhone 16 (NOTES.md "Backends beyond XNNPACK"): ALL lands
    # on the Neural Engine and wins by a wide margin, so the shipped variant
    # sets nothing and gets ALL. The knob stays per variant (as in deeplab /
    # ssdlite) because the sibling conv image-to-image model (style_transfer)
    # found the ANE pathological and pins CPU_AND_GPU.
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_unit=getattr(ct.ComputeUnit, variant.get("compute_unit") or "ALL"),
        compute_precision=precision,
    )
    program = to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
