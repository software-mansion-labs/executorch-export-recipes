import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_migan():
    key = "migan_xnnpack_vendored"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / "_migan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod



class _Fp16IO(torch.nn.Module):
    """fp32 I/O around an fp16 model, so the app contract stays fp32."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, image, mask):
        return self.inner(image.to(torch.float16), mask.to(torch.float16)).to(torch.float32)


def _fold_constants():
    """Constant-fold the edge program before partitioning.

    The fp16 build multiplies by Python scalars (lrelu_agc gain, clamp bounds,
    noise strength); the edge dialect lifts each one to an fp32 tensor and
    casts it to fp16 at run time: 60-odd portable `_to_dim_order_copy` ops per
    forward whose outputs enter the XNNPACK partition as 0-dim inputs.
    Folding them makes the scalars fp16 constants inside the partition and
    folds `noise_const * noise_strength` into one tensor.
    """
    from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
    from executorch.exir.passes.constant_prop_pass import constant_prop_pass

    class FoldConstants(ExportedProgramPassBase):
        def call(self, exported_program):
            return ExportedProgramPassResult(constant_prop_pass(exported_program), True)

    return FoldConstants()


def export(config: dict, out_dir: str) -> str:
    from torch.export import export as torch_export

    from exporter.backends import lower

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
    model = mg.InpaintingWrapper(generator)
    model.eval()

    inputs = (
        torch.rand(*image_shape, dtype=torch.float32),
        (torch.rand(*mask_shape) > 0.3).to(torch.float32),
    )

    if variant["quantized"]:
        # Checked first so an fp16 + quantized variant cannot slip through as
        # plain fp16.
        raise ValueError("int8 is not shipped for migan: wrong output on the RNE Android runtime, see NOTES.md")
    if variant["precision"] == "fp16":
        # fp16 is the only XNNPACK variant shipped: output matches fp32 to
        # ~1e-3 and ARM fp16 arithmetic makes it ~2x faster on device (S23);
        # the file is only 8% smaller. int8 was measured and dropped, see
        # NOTES.md "XNNPACK precision study".
        model = _Fp16IO(model.half()).eval()

    ep = torch_export(model, inputs, strict=True)
    program = lower(ep, config["backend"], transform_passes=[_fold_constants()])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
