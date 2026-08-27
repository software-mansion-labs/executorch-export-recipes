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
    "fcn_resnet50": (tvs.fcn_resnet50, lambda: tvs.FCN_ResNet50_Weights.DEFAULT),
    "fcn_resnet101": (tvs.fcn_resnet101, lambda: tvs.FCN_ResNet101_Weights.DEFAULT),
}


def _build(file_name: str):
    for arch in sorted(_ARCHS, key=len, reverse=True):
        if file_name.startswith(arch):
            ctor, weights = _ARCHS[arch]
            return ctor(weights=weights())
    raise ValueError(f"unknown fcn arch for variant file '{file_name}'")


def _lower(ep, precision: str):
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    prec = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[precision]
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18, compute_precision=prec
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
    program = _lower(ep, variant["precision"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
