import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "fast_sam_coreml_wrapper", _MODEL_DIR / "_fastsam_wrapper.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fast_sam_coreml_wrapper"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from ultralytics import FastSAM

    wrapper_mod = _load_wrapper()
    FastSAMInstanceSegmentation = wrapper_mod.FastSAMInstanceSegmentation

    size = config.get("size") or "s"
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])  # [1,3,640,640]
    method_size = shape[-1]
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]

    inner = FastSAM(f"FastSAM-{size}.pt").model
    model = FastSAMInstanceSegmentation(inner, method_size=method_size)
    model.eval()

    inputs = (torch.randn(*shape, dtype=torch.float32),)
    with torch.no_grad():
        model(*inputs)  # warmup: FastSAM head needs a forward to fix export-mode shapes

    ep = torch_export(model, inputs)
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )
    program = to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
