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
        "yolo26_pose_wrappers_coreml", pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_wrappers")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["yolo26_pose_wrappers_coreml"] = mod
    spec.loader.exec_module(mod)
    return mod


# NOTE: fp16 only. A FLOAT32 Core ML build of the YOLO26 graph forces the GPU
# path, where the MPSGraph MLIR compiler fails an internal assertion and aborts the
# process (see exporter_models/yolo26/coreml/export.py for the full write-up).
# Confirmed there on detection at 384/512/640; these heads share that backbone, so
# fp32 is not offered. fp16 is also the only precision the ANE accepts.
def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from ultralytics import YOLO

    YOLOPose = _load_wrapper().YOLOPose

    variant = config["variant"]
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]
    yolo = YOLO("yolo26n-pose.pt")
    model = YOLOPose(yolo.model).eval()

    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    inputs = (torch.randn(*shape, dtype=torch.float32),)
    with torch.no_grad():
        model(*inputs)  # warmup: fix export-mode head shapes
        ep = torch_export(model, inputs)

    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )
    program = to_edge_transform_and_lower(
        ep,
        partitioner=[CoreMLPartitioner(compile_specs=specs, lower_full_graph=False)],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
