"""Depth Anything V2 Small -> Qualcomm QNN (HTP) .pte, one for every supported
Snapdragon SoC (`exporter.qnn.SOCS`).

Same model and contract as the other backends: float32 [1, 3, 518, 518] RGB
normalized with ImageNet mean/std, raw relative depth [1, 518, 518]. fp16 on
the HTP, the whole graph in one QNN partition.

On an S26 Ultra, over the 145 NYUv2 test images (affine-aligned against GT,
RNE's stretch preprocessing): fp16 AbsRel 6.24% / delta1 95.67% against fp32
6.22% / 95.66%, at 20.4 ms per inference (shipped XNNPACK int8 and Vulkan fp16:
182 ms). Quantized variants are not shipped: a8w8 is 6.6 ms but 7.70% AbsRel,
a16w8 matches fp32 but takes 205 ms. Only the SM8850 context is device-validated.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch

from exporter import qnn


def _load_model():
    name = "depth_anything_v2_model_qnn"
    path = Path(__file__).resolve().parents[1] / "_model.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_model()


def export(config: dict, out_dir: str) -> str:
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError(
            "QNN_SDK_ROOT is not set -- the QNN AOT compiler needs the Qualcomm "
            "AI Engine Direct SDK (see skills/backends/qnn)."
        )

    from executorch.backends.qualcomm.utils.utils import (
        get_qnn_context_binary_alignment,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    variant = config["variant"]
    if variant["precision"] != "fp16":
        raise ValueError(f"only fp16 is exported for QNN, got {variant['precision']}")
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    model = _load_model()
    inputs = (torch.zeros(shape, dtype=torch.float32),)
    compiler_specs = qnn.compiler_specs(use_fp16=True)
    with torch.no_grad():
        traced = torch.export.export(model, inputs, strict=False).module()
        edge = to_edge_transform_and_lower_to_qnn(traced, inputs, compiler_specs)
    program = edge.to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=True, alloc_graph_output=True
            ),
            segment_alignment=get_qnn_context_binary_alignment(),
        )
    )

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    logging.info("wrote %s (%s)", out_path, variant["precision"])
    return str(out_path)
