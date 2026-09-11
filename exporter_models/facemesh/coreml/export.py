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
