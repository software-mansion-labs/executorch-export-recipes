import hashlib
import importlib.util
import sys
import urllib.request
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_WEIGHTS_DIR = _MODEL_DIR / "weights"


def _load_migan():
    key = "migan_coreml_vendored"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / "_migan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def _fetch_weights(mg, checkpoint: str) -> Path:
    path = _WEIGHTS_DIR / checkpoint
    if not path.exists():
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        url = mg.WEIGHTS_DRIVE_URL.format(file_id=mg.WEIGHTS_DRIVE_IDS[checkpoint])
        urllib.request.urlretrieve(url, path)

    expected = mg.WEIGHTS_SHA256[checkpoint]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"checkpoint '{checkpoint}' sha256 {actual} != expected {expected}; "
            f"delete {path} and re-run, or update WEIGHTS_SHA256 if upstream moved"
        )
    return path


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
    weights = _fetch_weights(mg, mg.CHECKPOINT_FOR_SIZE[size])
    generator.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model = mg.InpaintingWrapper(generator)
    model.eval()

    inputs = (
        torch.rand(*image_shape, dtype=torch.float32),
        (torch.rand(*mask_shape) > 0.3).to(torch.float32),
    )
    ep = torch_export(model, inputs, strict=True)
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[variant["precision"]]
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
        # compute_unit left at ExecuTorch's default (ALL). The sibling conv
        # image-to-image model (style_transfer) found the ANE pathological and
        # pins CPU_AND_GPU; fast_sam found the opposite. Not yet swept on an
        # iPhone for this graph (see NOTES.md) -- sweep before publishing.
    )
    program = to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
