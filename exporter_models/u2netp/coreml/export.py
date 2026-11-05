import importlib.util
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent.parent


def _load_wrapper():
    key = "u2netp_wrapper"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / "_wrapper.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    wrapper = _load_wrapper()
    model = wrapper.U2NetPForRNE(wrapper.load_model()).eval()
    inputs = (torch.rand(*shape, dtype=torch.float32),)
    ep = torch.export.export(model, inputs, strict=True)
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[variant["precision"]]
    specs = CoreMLBackend.generate_compile_specs(
        compute_unit=getattr(ct.ComputeUnit, variant["compute_unit"].upper()),
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )
    program = to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
    ).to_executorch()
    path = Path(out_dir) / variant["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(program.buffer)
    return str(path)
