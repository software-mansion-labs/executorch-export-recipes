"""YOLO26 detection -> Qualcomm QNN (HTP) .pte, one per Hexagon version.

Same Ultralytics model, wrapper and input contract as the xnnpack export
(letterboxed RGB in [0, 1]; boxes [300,4], scores [300], labels [300]), fp16 on
the HTP, and the target Hexagon version read from the variant file's suffix
(`..._v81.pte`).

TopK stays on the CPU (`skip_node_op_set`). The head takes the top 300 of 8400
anchors and then of 24000 scores, and on the HTP that costs ~3 ms of fp16
compute: on an S26 Ultra, 3.59 ms with TopK on CPU against 6.63 ms on the HTP,
at the same accuracy. Only TopK and the index gathers on its result run on the
CPU.

Accuracy on an S26 Ultra over 500 COCO val2017 images (mAP50-95, fp32 44.81):
fp16 44.77 at 3.59 ms. Quantized variants are not shipped: a16w8 is 44.42 and
slower (~4.0 ms), a8w8 is 35.13 at 1.95 ms (one 8-bit range cannot hold pixel
boxes and sigmoid scores).

Every size and resolution on v81 matches fp32 on an S26 Ultra (fork 1.4.1 runtime,
first 100 COCO val2017 images, mAP50-95 fp32 / QNN, latency per inference):

    384:  n 42.97/42.86 1.3 ms   s 52.41/52.21 1.9 ms   m 54.78/54.85 3.0 ms
          l 56.53/56.49 3.7 ms   x 61.46/61.58 7.2 ms
    512:  n 47.05/47.31 2.0 ms   s 56.24/56.08 2.5 ms   m 60.09/59.88 4.5 ms
          l 60.29/60.14 5.9 ms   x 64.73/64.12 12.1 ms
    640:  n 48.17/48.09 3.6 ms   s 56.69/56.91 4.1 ms   m 63.06/63.12 8.2 ms
          l 62.24/62.29 9.8 ms   x 65.73/65.47 21.9 ms

v69-v79 are compile-only. In the x86 HTP simulator (n, 20 images) v73-v79 match
v81 exactly; v69 differs per pixel and scores the same within 1 mAP.

Needs the software-mansion-labs 1.4.1 fork: without pytorch/executorch#24078
(Remove0DTensor keeps the selected element) the anchor stride lowering fails,
without #24057 (exact integer floor_divide) fp16 decodes wrong class indices,
without #22938 (DecomposeEmpty) the anchor `torch.full` leaves an `empty`
on the CPU that the delegate never binds, so the first inference crashes, and
without #22994 (BatchNorm folded into the conv) the 96 standalone fp16
BatchNorms inflate the scores: ~3 mAP lost at 384.
"""

import logging
import os
from pathlib import Path

import torch

# Hexagon version -> a SoC that carries it. Flagship parts only: fp16 on the HTP
# needs a premium-tier chip.
_HEXAGON_SOC = {
    "v69": "SM8450",  # Snapdragon 8 Gen 1
    "v73": "SM8550",  # Snapdragon 8 Gen 2
    "v75": "SM8650",  # Snapdragon 8 Gen 3
    "v79": "SM8750",  # Snapdragon 8 Elite
    "v81": "SM8850",  # Snapdragon 8 Elite Gen 5
}

_CPU_OPS = {"aten.topk.default"}


class _YoloDetection(torch.nn.Module):
    """Repo YOLOObjectDetection contract: packed [max_det,6] head -> (boxes, scores, labels)."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.inner.eval()
        self.inner.model[-1].export = True

    def forward(self, x):
        preds = self.inner(x)
        p = preds[0]
        return p[:, :4], p[:, 4], p[:, 5]


def _soc_for(file_name: str) -> str:
    version = Path(file_name).stem.rsplit("_", 1)[-1]
    if version not in _HEXAGON_SOC:
        raise ValueError(
            f"variant file '{file_name}' must end in a Hexagon version "
            f"{sorted(_HEXAGON_SOC)}; it is what the .pte is compiled for"
        )
    return _HEXAGON_SOC[version]


def export(config: dict, out_dir: str) -> str:
    # Checked before the imports below: without the SDK those fail deep in the
    # QNN AOT chain with an unrelated-looking error.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError(
            "QNN_SDK_ROOT is not set -- the QNN AOT compiler needs the Qualcomm "
            "AI Engine Direct SDK (see skills/backends/qnn)."
        )

    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_qnn_context_binary_alignment,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from ultralytics import YOLO

    size = config.get("size") or "n"
    variant = config["variant"]
    if variant["precision"] != "fp16":
        raise ValueError(f"only fp16 is exported for QNN, got {variant['precision']}")
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    soc = _soc_for(variant["file"])

    model = _YoloDetection(YOLO(f"yolo26{size}.pt").model).eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc),
        backend_options=generate_htp_compiler_spec(use_fp16=True),
    )
    edge = to_edge_transform_and_lower_to_qnn(
        model, inputs, compiler_specs, skip_node_op_set=_CPU_OPS
    )
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
    logging.info("wrote %s for %s (%s)", out_path, soc, variant["precision"])
    return str(out_path)
