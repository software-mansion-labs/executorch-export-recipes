from pathlib import Path

import torch
import torchvision.models.segmentation as tvs
from torch.export import export as torch_export


class _SegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The shipped .pte returns only the main "out" head (no aux).
        return self.model(x)["out"]


_ARCHS = {
    "deeplab_v3_mobilenet_v3_large": (
        tvs.deeplabv3_mobilenet_v3_large,
        lambda: tvs.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
    ),
    "deeplab_v3_resnet50": (
        tvs.deeplabv3_resnet50,
        lambda: tvs.DeepLabV3_ResNet50_Weights.DEFAULT,
    ),
    "deeplab_v3_resnet101": (
        tvs.deeplabv3_resnet101,
        lambda: tvs.DeepLabV3_ResNet101_Weights.DEFAULT,
    ),
}


def _build(file_name: str):
    for arch in sorted(_ARCHS, key=len, reverse=True):
        if file_name.startswith(arch):
            ctor, weights = _ARCHS[arch]
            return ctor(weights=weights())
    raise ValueError(f"unknown deeplab arch for variant file '{file_name}'")


def _lower(ep, precision: str, compute_unit: str | None = None):
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    prec = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[precision]
    # generate_compile_specs defaults compute_unit to ALL. The mobilenet-v3-large
    # backbone is the smallest graph in this family and the only model in the
    # whole iOS estate where Core ML does not beat XNNPACK (30.08 ms against
    # 29.39 ms on an iPhone 17), which is the signature of ANE dispatch overhead
    # on a small graph rather than of slow compute. Swept on device; see
    # docs/ssdlite320-coreml-compute-units.md.
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_unit=getattr(ct.ComputeUnit, compute_unit or "ALL"),
        compute_precision=prec,
    )
    # lower_full_graph=False keeps anything Core ML cannot take on the CPU
    # backend instead of failing the export outright.
    return to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs, lower_full_graph=False)]
    ).to_executorch()


def export(config: dict, out_dir: str) -> str:
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    inputs = (torch.randn(shape, dtype=torch.float32),)

    model = _SegWrapper(_build(variant["file"])).eval()
    ep = torch_export(model, inputs)
    program = _lower(ep, variant["precision"], variant.get("compute_unit"))

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
