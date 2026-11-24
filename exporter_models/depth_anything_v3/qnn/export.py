"""Depth Anything 3 Small -> Qualcomm QNN (HTP) .pte, one for every supported
Snapdragon SoC (`exporter.qnn.SOC_GROUPS`).

Same contract and export adaptations as the Vulkan export (static RoPE extent,
eager attention, positional constants folded after functionalization; reused
from `../vulkan/export.py`): float32 [1, 3, 518, 518] RGB normalized with
ImageNet mean/std, raw depth [1, 518, 518]. fp16 on the HTP, one QNN
partition.

On an S26 Ultra, over the 145 NYUv2 test images (affine-aligned against GT,
RNE's stretch preprocessing): fp16 AbsRel 8.50% / delta1 92.25% against fp32
8.49% / 92.30%, at 42.5 ms per inference (shipped Vulkan fp16: 236 ms). a8w8
fails QNN context binary generation; Depth Anything V2's a8w8 lost 1.5 AbsRel
points and its a16w8 was 10x slower than fp16, so quantization was not pursued.
Only the SM8850 context is device-validated.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch

from exporter import qnn

_DIR = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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

    model = _load(_DIR / "_model.py", "depth_anything_v3_model_qnn").build_model()
    vulkan = _load(_DIR / "vulkan" / "export.py", "depth_anything_v3_vulkan_for_qnn")
    from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D

    for layer in model.modules():
        if hasattr(layer, "fused_attn"):
            layer.fused_attn = False
        if isinstance(layer, RotaryPositionEmbedding2D):
            layer._export_max_position = max(shape[-2:]) // 14 + 1
            layer.forward = vulkan._static_rope.__get__(layer)

    inputs = (torch.zeros(shape, dtype=torch.float32),)
    compiler_specs = qnn.compiler_specs(variant, use_fp16=True)
    with torch.no_grad():
        graph = (
            torch.export.export(model, inputs, strict=False)
            .run_decompositions({})
            .module()
        )
        vulkan._fold_constants(graph)
        traced = torch.export.export(graph, inputs, strict=False).module()
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
