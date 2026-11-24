"""RF-DETR nano detection -> Qualcomm QNN (HTP) .pte, one for every supported
Snapdragon SoC (`exporter.qnn.SOCS`).

Same model, patches, wrapper and input contract as the vulkan export (RGB
normalized with ImageNet mean/std, [1,3,384,384]; boxes [300,4] in pixels,
scores [300], labels [300]), fp16 on the HTP. The whole graph is one
QNN partition.

On an S26 Ultra over 500 COCO val2017 images (mAP50-95, fp32 50.08): fp16 50.05
at 9.8 ms per inference. Only the SM8850 context is device-validated. Quantized
variants are not shipped (all need pytorch/executorch#24384 and #24431): a16w8
is 49.81 at 10.8 ms, a16w8 with an a8w8 backbone 46.46 at 6.7 ms, a8w8 encoder
layers only 46.49 at 6.7 ms, a8w8 layers 6-11 only 48.85 at 11.3 ms, and a16w4
8.55 at 10.6 ms. a8w8 everywhere loses the decoder (34.1 on a CPU check of 50
images).

Needs the software-mansion-labs 1.4.1 fork: without pytorch/executorch#24382
(DecomposeMinMaxDim matches the op, not torch.max source partitions) the
`max(dim)` labels fail context binary generation, and without #24383 (drop
single-chunk split_with_sizes) the graph splits into two partitions around a
CPU split.
"""

import importlib.util
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from exporter import qnn

_MODEL_DIR = Path(__file__).resolve().parent.parent


def _apply_patches():
    spec = importlib.util.spec_from_file_location(
        "rfdetr_det_patches", _MODEL_DIR / "_patches.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_patches()


class _RFDETRDetectionWrapper(nn.Module):
    def __init__(self, model: nn.Module, resolution: int):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def forward(self, x: torch.Tensor):
        pred_boxes, pred_logits = self.model(x)
        scores, labels = pred_logits[0].sigmoid().max(dim=-1)
        boxes = pred_boxes[0]
        cxy = boxes[..., :2]
        wh = boxes[..., 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        bboxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution
        return bboxes, scores, labels.float()


def export(config: dict, out_dir: str) -> str:
    # Checked before the imports below: without the SDK those fail deep in the
    # QNN AOT chain with an unrelated-looking error.
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

    _apply_patches()
    from rfdetr import RFDETRNano

    model = RFDETRNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    wrapper = _RFDETRDetectionWrapper(actual, model.model.resolution).eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    compiler_specs = qnn.compiler_specs(use_fp16=True)
    with torch.no_grad():
        # Non-strict first: the patches module is loaded by path, which a strict
        # (dynamo) trace cannot import.
        traced = torch.export.export(wrapper, inputs, strict=False).module()
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
