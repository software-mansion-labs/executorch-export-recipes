"""MobileNetV2 -> Qualcomm QNN (HTP) .pte.

First proof-of-concept for the ExecuTorch Qualcomm backend: a small torchvision
classifier that quantizes cleanly to 8a8w on the Hexagon Tensor Processor, and
that upstream ExecuTorch ships a reference runner for (so the produced .pte can
be cross-checked against known top-1/top-5). See NOTES.md and
`skills/backends/qnn` for the host setup -- the AOT compile needs an x86_64
Linux host with the QNN SDK, it does NOT run on macOS.

Two knobs that are not part of the model spec come from the environment,
because a QNN .pte is compiled for one specific SoC:

    QNN_SOC_MODEL        target chipset, a QcomChipset member (default SM8650)
    QNN_CALIBRATION_DIR  folder of images for int8 PTQ calibration
    QNN_CALIBRATION_SIZE how many of them to use (default 100)
"""

import logging
import os
import random
from pathlib import Path

import torch

_DEFAULT_SOC = "SM8650"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# torchvision's IMAGENET1K_V2 eval preprocessing is resize-256 -> center-crop-224
# for MobileNetV2; keep the ratio if the config ever asks for another size.
_RESIZE_RATIO = 256 / 224

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class _MobileNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # The shipped .pte normalizes the raw [0,1] image inside the graph, so
        # the runtime feeds a plain RGB tensor (same contract as efficientnet).
        return self.model((x - self.mean) / self.std)


def _calibration_inputs(shape) -> list:
    """Images used to calibrate activation ranges for 8a8w PTQ.

    Calibration is what makes int8 accurate, so point QNN_CALIBRATION_DIR at a
    few hundred real ImageNet validation images (imagenet-mini/val does fine).
    Without it we calibrate on noise: the export succeeds and runs on the HTP,
    but the predictions are worthless -- enough to smoke-test the toolchain, not
    to judge the model.
    """
    size = shape[-1]
    directory = os.environ.get("QNN_CALIBRATION_DIR")
    if not directory:
        logging.warning(
            "QNN_CALIBRATION_DIR is unset: calibrating int8 on RANDOM inputs. "
            "The .pte will export and run, but its predictions will be poor."
        )
        return [(torch.rand(shape),) for _ in range(8)]

    from PIL import Image
    from torchvision import transforms as T

    preprocess = T.Compose(
        [
            T.Resize(round(size * _RESIZE_RATIO)),
            T.CenterCrop(size),
            T.ToTensor(),  # [0,1]; the graph does the mean/std itself
        ]
    )
    paths = sorted(
        p for p in Path(directory).rglob("*") if p.suffix.lower() in _IMAGE_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"no images under QNN_CALIBRATION_DIR={directory}")
    random.Random(0).shuffle(paths)
    paths = paths[: int(os.environ.get("QNN_CALIBRATION_SIZE", "100"))]
    logging.info("calibrating on %d images from %s", len(paths), directory)
    return [(preprocess(Image.open(p).convert("RGB")).unsqueeze(0),) for p in paths]


def export(config: dict, out_dir: str) -> str:
    # Checked before the imports below: without the SDK those fail deep in the
    # QNN AOT chain with an unrelated-looking error.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError(
            "QNN_SDK_ROOT is not set -- the QNN AOT compiler needs the Qualcomm "
            "AI Engine Direct SDK (see skills/backends/qnn)."
        )

    import torchvision.models as tvm
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
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    soc = os.environ.get("QNN_SOC_MODEL", _DEFAULT_SOC)
    if not hasattr(QcomChipset, soc):
        valid = sorted(c.name for c in QcomChipset)
        raise ValueError(f"unknown QNN_SOC_MODEL '{soc}'; QcomChipset has {valid}")

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    quantized = variant["quantized"]

    model = _MobileNetWrapper(
        tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2)
    ).eval()
    sample_input = (torch.rand(shape, dtype=torch.float32),)

    # fp16 is an HTP compile-time precision (premium SoCs only); int8 is real
    # 8a8w PTQ done here, before lowering.
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc),
        backend_options=generate_htp_compiler_spec(use_fp16=not quantized),
    )

    if quantized:
        captured = torch.export.export(model, sample_input, strict=False).module()
        annotated = prepare_pt2e(
            captured, make_quantizer(quant_dtype=QuantDtype.use_8a8w, soc_model=soc)
        )
        with torch.no_grad():
            for calib_input in _calibration_inputs(shape):
                annotated(*calib_input)
        to_lower = convert_pt2e(annotated)
    else:
        to_lower = model

    edge = to_edge_transform_and_lower_to_qnn(to_lower, sample_input, compiler_specs)
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
