"""EfficientNetV2-S -> Qualcomm QNN (HTP) .pte, one context binary per Hexagon version.

Same contract as the xnnpack export: the graph takes a raw [0,1] image and does
the ImageNet normalization itself, because RNE passes `{alpha: 1/255}` for this
model rather than IMAGENET_NORM.

A QNN .pte is compiled for one Hexagon version, so unlike the other backends each
variant file carries the target as a suffix (`..._v81.pte`) and this script reads
it back from the filename. One representative SoC per version is enough: the
device-side runtime keys off the Hexagon version, not the chipset.

Calibration needs ImageNet-like images, which torchvision cannot fetch the way it
fetches VOC. Point QNN_CALIBRATION_DIR at any ImageFolder-shaped directory of
JPEGs; ImageNetV2 matched-frequency works and is what these were calibrated on:

    https://huggingface.co/datasets/vaishaal/ImageNetV2 (imagenetv2-matched-frequency.tar)

    QNN_CALIBRATION_DIR   directory of calibration images (required)
    QNN_CALIBRATION_SIZE  how many to use (default 100)

a16w8 is the default. a8w8, with a histogram activation observer, is 1.75x faster
for 1.4 top-1 on an S26 Ultra over 500 ImageNetV2 images: 70.40% at 1.34 ms against
71.80% at 2.34 ms. Min/max is wrong at 8 bits here: outlier activations stretch the
range until the logits carry nothing (0% top-1). Only v81 is device-validated.
"""

import logging
import os
import random
from pathlib import Path

import torch
import torchvision.models as tvm
from torch.export import export as torch_export

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_DEFAULT_CALIBRATION_SIZE = 100

# Hexagon version -> a SoC that carries it.
_HEXAGON_SOC = {
    "v69": "SM8450",  # Snapdragon 8 Gen 1
    "v73": "SM8550",  # Snapdragon 8 Gen 2
    "v75": "SM8650",  # Snapdragon 8 Gen 3
    "v79": "SM8750",  # Snapdragon 8 Elite
    "v81": "SM8850",  # Snapdragon 8 Elite Gen 5
}


class _EfficientNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # The shipped .pte normalizes the raw [0,1] image inside the graph.
        return self.model((x - self.mean) / self.std)


def _soc_for(file_name: str) -> str:
    version = Path(file_name).stem.rsplit("_", 1)[-1]
    if version not in _HEXAGON_SOC:
        raise ValueError(
            f"variant file '{file_name}' must end in a Hexagon version "
            f"{sorted(_HEXAGON_SOC)}; it is what the .pte is compiled for"
        )
    return _HEXAGON_SOC[version]


def _calibrate(prepared, size: int, resize: list):
    """Activation ranges for a16w8 PTQ, from real photographs.

    Fed in exactly the form the graph expects at runtime: a raw [0,1] image at
    the variant's input size, with the normalization left to the model.
    """
    import torchvision.transforms.functional as F
    from torchvision.io import decode_image

    root = os.environ.get("QNN_CALIBRATION_DIR")
    if not root or not Path(root).is_dir():
        raise RuntimeError(
            "QNN_CALIBRATION_DIR must point at a directory of calibration images "
            "(see the module docstring for where to get ImageNetV2)."
        )
    images = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    if len(images) < size:
        raise RuntimeError(f"found {len(images)} images under {root}, need {size}")

    random.seed(0)
    logging.info("calibrating on %d images from %s", size, root)
    with torch.no_grad():
        for path in random.sample(images, k=size):
            img = decode_image(str(path))
            # a handful of ImageNet files are greyscale or carry an alpha channel
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            img = img[:3].unsqueeze(0)
            img = F.resize(
                img, resize, interpolation=F.InterpolationMode.BILINEAR, antialias=True
            )
            prepared(F.convert_image_dtype(img, torch.float))


_QUANT_DTYPE = {"a16w8": "use_16a8w", "a8w8": "use_8a8w"}


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
    from torchao.quantization.pt2e.observer import HistogramObserver, MinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    soc = _soc_for(variant["file"])
    quantized = variant["quantized"]
    a8w8 = variant["precision"] == "a8w8"
    inputs = (torch.randn(shape, dtype=torch.float32),)

    model = _EfficientNetWrapper(
        tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.DEFAULT)
    ).eval()

    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc),
        backend_options=generate_htp_compiler_spec(use_fp16=not quantized),
    )

    if quantized:
        prepared = prepare_pt2e(
            torch_export(model, inputs, strict=False).module(),
            # a16w8: MinMaxObserver rather than make_quantizer's
            # MovingAverageMinMaxObserver default: the moving average trails the
            # real activation range. See the note in the deeplab_v3 qnn export,
            # where it was worth 1.59 mIoU on device. a8w8: histogram, see the
            # module docstring.
            make_quantizer(
                quant_dtype=getattr(QuantDtype, _QUANT_DTYPE[variant["precision"]]),
                soc_model=soc,
                act_observer=HistogramObserver if a8w8 else MinMaxObserver,
                per_channel_linear=a8w8,
            ),
        )
        _calibrate(
            prepared,
            int(os.environ.get("QNN_CALIBRATION_SIZE", _DEFAULT_CALIBRATION_SIZE)),
            list(shape[-2:]),
        )
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
