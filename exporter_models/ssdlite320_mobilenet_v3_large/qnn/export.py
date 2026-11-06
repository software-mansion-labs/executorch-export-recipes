"""SSDLite320 MobileNetV3 -> Qualcomm QNN (HTP) .pte, one per Hexagon version.

Same vendored torchvision model, wrapper and input contract as the xnnpack
export (RGB in [0, 1], CHW [3,320,320]; boxes [300,4], scores [300], labels
[300]), fp16 on the HTP, and the target Hexagon version read from the variant
file's suffix (`..._v81.pte`). TopK stays on the CPU (`skip_node_op_set`), as in
the YOLO26 QNN export.

The in-graph NMS computes IoU on boxes scaled to [0, 1] (_model/ssd.py): in
pixels a 316x285 box has an area of 90k, which overflows fp16, and the HTP then
reports NaN IoUs as overlaps and suppresses every box but the first.

On an S26 Ultra over 500 COCO val2017 images (mAP50-95, fp32 22.40): fp16
22.32 at 4.2 ms per inference (17.92 before the IoU fix). Quantized variants
are not shipped: a16w8 is 22.17 at 4.3 ms, a8w8 is 21.04 at 3.0 ms, and a8w8
with decode + NMS left in fp32 on the CPU is 21.86 at 4.0 ms.
Only v81 is device-validated; v69-v79 target SoCs we do not have.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent.parent

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


def _load_model_module():
    pkg_init = _MODEL_DIR / "_model" / "__init__.py"
    module_name = f"_vendored_{_MODEL_DIR.name}_qnn_model"
    spec = importlib.util.spec_from_file_location(
        module_name, pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_model")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class _SsdliteWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        det = self.model([image])[0]
        return det["boxes"], det["scores"], det["labels"].float()


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

    variant = config["variant"]
    if variant["precision"] != "fp16":
        raise ValueError(f"only fp16 is exported for QNN, got {variant['precision']}")
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    soc = _soc_for(variant["file"])

    m = _load_model_module()
    base_model = m.ssdlite320_mobilenet_v3_large(
        weights=m.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    ).eval()
    model = _SsdliteWrapper(base_model).eval()
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
