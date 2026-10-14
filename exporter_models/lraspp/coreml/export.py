from pathlib import Path

import torch
import torchvision.models.segmentation as tvs
from torch.export import export as torch_export


def _ane_argmax(logits, k: int):
    """argmax(dim=1) built only from ops the ANE runs.

    torch.argmax lowers to ios18.reduce_argmax, which has no ANE kernel and is
    pinned to the CPU: the ANE then writes all K channels out for the CPU to
    reduce, so the transfer is still paid and a multi-million element reduction
    is added on top (31 -> 65 ms on an iPhone 16, measured). reduce_max and
    elementwise ops are ANE-resident, so the same result is built from those
    (25 ms, and 200 of 201 ops stay on the ANE).

    torch.argmax returns the FIRST maximal index while (mask * arange).amax
    returns the last, hence the reversal. Verified pixel-exact on 520x520.
    """
    ar = torch.arange(k, dtype=logits.dtype, device=logits.device).view(1, k, 1, 1)
    peak = logits.amax(dim=1, keepdim=True)
    hit = (logits == peak).to(logits.dtype)
    return (k - 1) - (((k - 1) - ar) * hit).amax(dim=1)


class _SegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The shipped .pte returns only the main "out" head (no aux).
        logits = self.model(x)["out"]
        return _ane_argmax(logits, 21).to(torch.int32)


def _build(file_name: str):
    return tvs.lraspp_mobilenet_v3_large(
        weights=tvs.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    )


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
