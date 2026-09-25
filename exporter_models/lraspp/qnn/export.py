"""LR-ASPP MobileNetV3 -> Qualcomm QNN (HTP) .pte, one per Hexagon version.

Same torchvision model and input contract as the xnnpack export
(already ImageNet-normalized, RNE passes IMAGENET_NORM), and the same recipe as
the DeepLabV3 QNN export: a16w8 PTQ with min/max activation observers, a class
index per pixel as output, and the target Hexagon version read from the variant
file's suffix (`..._v81.pte`).

    QNN_CALIBRATION_SIZE  images used for a16w8 PTQ calibration (default 100)

eps is 1e-5, not QNN's 16-bit default: the stage 5-6 depthwise convs have dead
channels that still carry a folded bias (up to 0.93), and the int32 bias scale
s_in * s_w is so small at a floored s_w that the bias saturates to ~0. That cost
3.6 mIoU: on an S26 Ultra 61.48% at the default, 65.83% at 1e-5 (XNNPACK int8
65.06%, fp32 65.76%), 2.06 ms.
"""

import logging
import os
import random
from pathlib import Path

import torch
import torchvision.models.segmentation as tvs
from torch.export import export as torch_export

_RESIZE = [520, 520]
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_DEFAULT_CALIBRATION_SIZE = 100

# Hexagon version -> a SoC that carries it. Flagship parts only: fp16 on the HTP
# needs a premium-tier chip, and these all have the same 8 MB VTCM budget.
_HEXAGON_SOC = {
    "v69": "SM8450",  # Snapdragon 8 Gen 1
    "v73": "SM8550",  # Snapdragon 8 Gen 2
    "v75": "SM8650",  # Snapdragon 8 Gen 3
    "v79": "SM8750",  # Snapdragon 8 Elite
    "v81": "SM8850",  # Snapdragon 8 Elite Gen 5
}


class _IndexMapWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The "out" head, reduced to a class index per pixel:
        # (1,520,520) int32 instead of (1,21,520,520) float32, which the QNN
        # argmax builder keeps inside the delegate. See RNE issue #1459.
        return torch.argmax(self.model(x)["out"], dim=1).to(torch.int32)


def _soc_for(file_name: str) -> str:
    version = Path(file_name).stem.rsplit("_", 1)[-1]
    if version not in _HEXAGON_SOC:
        raise ValueError(
            f"variant file '{file_name}' must end in a Hexagon version "
            f"{sorted(_HEXAGON_SOC)}; it is what the .pte is compiled for"
        )
    return _HEXAGON_SOC[version]


def _calibrate(prepared):
    """Activation ranges for a16w8 PTQ, from VOC segmentation training images.

    The same source the xnnpack int8 export calibrates on, fed through the same
    preprocessing the model expects at runtime (resize to 520, ImageNet norm).
    """
    import torchvision.transforms.functional as F
    from torchvision import datasets
    from torchvision.io import decode_image

    random.seed(0)
    ds = datasets.VOCSegmentation(
        root="datasets/VOCSegmentation", year="2012", image_set="train", download=True
    )
    size = int(os.environ.get("QNN_CALIBRATION_SIZE", _DEFAULT_CALIBRATION_SIZE))
    logging.info("calibrating on %d VOC images", size)
    with torch.no_grad():
        for img_path in random.sample(ds.images, k=size):
            img = decode_image(img_path).unsqueeze(0)
            img = F.resize(
                img, _RESIZE, interpolation=F.InterpolationMode.BILINEAR, antialias=True
            )
            img = F.convert_image_dtype(img, torch.float)
            prepared(F.normalize(img, mean=_IMAGENET_MEAN, std=_IMAGENET_STD))


def export(config: dict, out_dir: str) -> str:
    # Checked before the imports below: without the SDK those fail deep in the
    # QNN AOT chain with an unrelated-looking error.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError(
            "QNN_SDK_ROOT is not set -- the QNN AOT compiler needs the Qualcomm "
            "AI Engine Direct SDK (see skills/backends/qnn)."
        )

    from executorch.backends.qualcomm.export_utils import make_quantizer
    from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_qnn_context_binary_alignment,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from torchao.quantization.pt2e.observer import MinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    soc = _soc_for(variant["file"])
    quantized = variant["quantized"]
    inputs = (torch.randn(shape, dtype=torch.float32),)

    model = _IndexMapWrapper(
        tvs.lraspp_mobilenet_v3_large(
            weights=tvs.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        )
    ).eval()

    # fp16 is an HTP compile-time precision, not a quantization pass; a16w8 is
    # 16-bit activations over 8-bit weights, done here before lowering.
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc),
        backend_options=generate_htp_compiler_spec(use_fp16=not quantized),
    )

    if quantized:
        prepared = prepare_pt2e(
            torch_export(model, inputs, strict=False).module(),
            # MinMaxObserver, not the make_quantizer default of
            # MovingAverageMinMaxObserver, whose moving average trails the real
            # activation range: it cost DeepLabV3 1.59 mIoU.
            make_quantizer(
                quant_dtype=QuantDtype.use_16a8w,
                soc_model=soc,
                act_observer=MinMaxObserver,
                eps=1e-5,
            ),
        )
        _calibrate(prepared)
        model = convert_pt2e(prepared)

    edge = to_edge_transform_and_lower_to_qnn(model, inputs, compiler_specs)
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
