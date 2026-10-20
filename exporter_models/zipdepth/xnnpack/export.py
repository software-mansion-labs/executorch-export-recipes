import importlib.util
import sys
from pathlib import Path

import torch


def _load_model():
    name = "zipdepth_model_xnnpack"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_model()


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    torch.set_num_threads(4)
    model = _load_model()
    inputs = (torch.zeros(shape, dtype=torch.float32),)
    with torch.no_grad():
        ep = torch.export.export(model, inputs, strict=False)
    program = to_edge_transform_and_lower(
        ep, partitioner=[XnnpackPartitioner(extended_ops=True)]
    ).to_executorch()
    path = Path(out_dir) / variant["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(program.buffer)
    return str(path)
