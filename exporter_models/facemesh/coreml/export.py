import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_PKG = _MODEL_DIR / "_facemesh"


def _load_facemesh():
    spec = importlib.util.spec_from_file_location(
        "facemesh_pkg_coreml",
        _PKG / "__init__.py",
        submodule_search_locations=[str(_PKG)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["facemesh_pkg_coreml"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from torch.export import export as torch_export

    fm = _load_facemesh()
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    model = fm.build()
    inputs = (torch.randn(*shape, dtype=torch.float32),)
    ep = torch_export(model, inputs, strict=True)

    precision = {
        "fp16": ct.precision.FLOAT16,
        "fp32": ct.precision.FLOAT32,
    }[variant["precision"]]
    # ALL, stated rather than defaulted. It is also the default, but the right
    # compute unit for this graph is a measurement and not an assumption: on an
    # iPhone 16 Pro, ALL (0.610 ms) tracks CPU_AND_NE (0.604 ms), so it does
    # reach the Neural Engine here, while CPU_ONLY costs 2.15x-vs-2.52x and
    # CPU_AND_GPU (2.343 ms) is the one arm slower than plain XNNPACK. Anyone
    # re-tuning this should re-measure rather than trust the default. See the
    # compute-unit table in NOTES.md.
    specs = CoreMLBackend.generate_compile_specs(
        compute_unit=ct.ComputeUnit.ALL,
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
