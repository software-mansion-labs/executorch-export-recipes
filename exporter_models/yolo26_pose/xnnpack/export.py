import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_wrapper():
    pkg_init = _MODEL_DIR / "_wrappers" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "yolo26_pose_wrappers", pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_wrappers")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["yolo26_pose_wrappers"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from ultralytics import YOLO

    YOLOPose = _load_wrapper().YOLOPose

    variant = config["variant"]
    yolo = YOLO("yolo26n-pose.pt")
    model = YOLOPose(yolo.model).eval()

    methods = {}
    for method_name, md in variant["methods"].items():
        shape = tuple(md["inputs"][0]["shape"])  # [1,3,R,R]
        inputs = (torch.randn(*shape, dtype=torch.float32),)
        with torch.no_grad():
            model(*inputs)  # warmup
            methods[method_name] = torch_export(model, inputs)

    program = to_edge_transform_and_lower(
        methods, partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
