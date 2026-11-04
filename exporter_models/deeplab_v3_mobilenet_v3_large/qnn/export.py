"""DeepLabV3 -> Qualcomm QNN (HTP) .pte, one context binary per Hexagon version.

Same three torchvision architectures as the xnnpack export, and the same
contract: input is already ImageNet-normalized (RNE passes IMAGENET_NORM), and
only the main "out" head is returned.

A QNN .pte is compiled for one Hexagon version, so unlike the other backends
each variant file carries the target as a suffix (`..._v81.pte`) and this script
reads it back from the filename. One representative SoC per version is enough:
the device-side runtime keys off the Hexagon version, not the chipset, and every
SoC listed here has 8 MB of VTCM.

    QNN_CALIBRATION_SIZE  images used for PTQ calibration (default 100)

No fp16 variants are published. fp16 hangs the HTP on this model: the ASPP's
dilated convolution over a 65x65 input trips a watchdog after ~11.5 s with QNN
error 1007, and no compiler option avoids it (pytorch/executorch#23096). The
fp16 path below is kept because the failure is specific to that layer shape, but
config.json declares a8w8 only.

All variants are a8w8: at the same mIoU it halves latency on an S26
Ultra (200 VOC2012 val images; a16w8 -> a8w8): ResNet50 76.74% / 20.4 ms ->
76.44% / 9.9 ms, ResNet101 78.90% / 29.4 ms -> 78.96% / 14.9 ms, MobileNetV3
69.02% / 2.98 ms -> 69.14% / 1.34 ms. MobileNetV3 needs a histogram activation
observer at 8 bits; with min/max one outlier stretches the range and costs ~11 mIoU.
Only v81 is device-validated; v69-v79 target SoCs we do not have.
In the x86 HTP simulator v73-v79 match v81; v69 differs per pixel but scores the
same over 200 VOC images (LR-ASPP a8w8 65.62%, a16w8 65.69%).
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

_ARCHS = {
    "deeplab_v3_mobilenet_v3_large": (
        tvs.deeplabv3_mobilenet_v3_large,
        lambda: tvs.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
    ),
    "deeplab_v3_resnet50": (
        tvs.deeplabv3_resnet50,
        lambda: tvs.DeepLabV3_ResNet50_Weights.DEFAULT,
    ),
    "deeplab_v3_resnet101": (
        tvs.deeplabv3_resnet101,
        lambda: tvs.DeepLabV3_ResNet101_Weights.DEFAULT,
    ),
}

# Hexagon version -> a SoC that carries it. Flagship parts only: fp16 on the HTP
# needs a premium-tier chip, and these all have the same 8 MB VTCM budget.
_HEXAGON_SOC = {
    "v69": "SM8450",  # Snapdragon 8 Gen 1
    "v73": "SM8550",  # Snapdragon 8 Gen 2
    "v75": "SM8650",  # Snapdragon 8 Gen 3
    "v79": "SM8750",  # Snapdragon 8 Elite
    "v81": "SM8850",  # Snapdragon 8 Elite Gen 5
}


class _DeeplabWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Only the main "out" head (no aux), reduced to a class index per pixel:
        # (1,520,520) int32 instead of (1,21,520,520) float32, 22.71 MB -> 1.08 MB
        # across the DSP boundary every inference. See RNE issue #1459.
        #
        # Plain torch.argmax is right here, unlike the Core ML exports: QNN has a
        # native builder for aten.argmax.default, so the reduction lowers into the
        # same delegate rather than falling back to a portable CPU op.
        return torch.argmax(self.model(x)["out"], dim=1).to(torch.int32)


class _SplitInputConv(torch.nn.Module):
    """A conv computed as the sum of convs over slices of its input channels.

    Exact (convolution is linear in the input channels) and needed for the
    ResNet ASPP: its dilated 3x3 convs read 2048 channels at 65x65, and in a16w8
    the HTP cannot tile them into the 8 MiB VTCM ("not sufficiently tiled to fit
    in TCM"). The dilation-36 kernel spans 73 pixels, wider than the map, so it
    cannot be tiled spatially and a whole input slice must fit: 1024 channels
    are 1024*65*65*2 B = 8.65 MB and still fail, 512 are 4.33 MB.

    ResNet50 a16w8 v81 on an S26 Ultra, 200 VOC2012 val images: 76.74% mIoU at
    20.1 ms, against 76.71% for the published XNNPACK int8 file.
    """

    def __init__(self, conv: torch.nn.Conv2d, chunk: int):
        super().__init__()
        assert conv.groups == 1
        self.convs = torch.nn.ModuleList()
        for i, w in enumerate(conv.weight.split(chunk, dim=1)):
            # the bias is added once, by the first slice
            has_bias = i == 0 and conv.bias is not None
            part = torch.nn.Conv2d(
                w.shape[1],
                conv.out_channels,
                conv.kernel_size,
                padding=conv.padding,
                dilation=conv.dilation,
                bias=has_bias,
            )
            part.weight = torch.nn.Parameter(w.detach().clone())
            if has_bias:
                part.bias = torch.nn.Parameter(conv.bias.detach().clone())
            self.convs.append(part)
        self.chunk = chunk

    def forward(self, x):
        parts = x.split(self.chunk, dim=1)
        out = self.convs[0](parts[0])
        for conv, part in zip(self.convs[1:], parts[1:]):
            out = out + conv(part)
        return out


# Split only ASPPs wider than this; MobileNetV3's reads 960 channels (8.1 MB,
# fits) and is left untouched.
_ASPP_MAX_CHANNELS = 1024
_ASPP_SLICE_CHANNELS = 512


def _split_wide_aspp_convs(model):
    for module in model.modules():
        if not isinstance(module, tvs.deeplabv3.ASPPConv):
            continue
        conv, bn = module[0], module[1]
        if conv.in_channels > _ASPP_MAX_CHANNELS:
            # Fold the BatchNorm first: after the split it would follow an add,
            # not a conv, and prepare_pt2e would no longer fold it.
            fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv.eval(), bn.eval())
            module[0] = _SplitInputConv(fused, _ASPP_SLICE_CHANNELS)
            module[1] = torch.nn.Identity()
    return model


def _arch_for(file_name: str):
    # longest match first so resnet101 isn't shadowed by resnet50, etc.
    for arch in sorted(_ARCHS, key=len, reverse=True):
        if file_name.startswith(arch):
            return _ARCHS[arch]
    raise ValueError(f"unknown deeplab arch for variant file '{file_name}'")


# ResNet-101 carries conv channels that BatchNorm has all but zeroed (folded
# weights down to ~1e-8). QNN's a16w8 default eps (1.5e-9) lets those channels
# keep a per-channel weight scale a million times below their neighbours', and
# the HTP computes them as garbage: on an S26 Ultra one such layer1 channel came
# out 338x too large and the whole model scored 0.1% mIoU. Flooring scales at
# QNN's 8-bit default (3.9e-7) fixes it (layer1 cosine vs fp32: 0.67 -> 0.9996).
# ResNet-101 then scores 78.90% mIoU at 29.7 ms (XNNPACK int8: 78.82%).
#
# MobileNetV3 needs more: its stage 5-6 depthwise convs have dead channels that
# still carry a folded bias (up to 0.93), and the int32 bias scale s_in * s_w is
# so small at a floored s_w that the bias saturates to ~0. 3.9e-7 is not enough;
# 1e-5 is: 66.55% -> 69.02% mIoU on an S26 Ultra (XNNPACK int8 66.56%, fp32 69.28%).
# ResNet-50 keeps the default, matching its published files.
_WEIGHT_EPS = {
    "deeplab_v3_resnet101": 3.921568627450981e-07,  # qconfig.DEFAULT_EPS_8BIT
    "deeplab_v3_mobilenet_v3_large": 1e-5,
}


_QUANT_DTYPE = {"a16w8": "use_16a8w", "a8w8": "use_8a8w"}


def _eps_for(file_name: str):
    for arch, eps in _WEIGHT_EPS.items():
        if file_name.startswith(arch):
            return eps
    return None


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
    from torchao.quantization.pt2e.observer import HistogramObserver, MinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    builder, weights = _arch_for(variant["file"])
    soc = _soc_for(variant["file"])
    quantized = variant["quantized"]
    histogram = (
        variant["precision"] == "a8w8" and variant["size"] == "mobilenet_v3_large"
    )
    inputs = (torch.randn(shape, dtype=torch.float32),)

    model = _DeeplabWrapper(_split_wide_aspp_convs(builder(weights=weights()))).eval()

    # fp16 is an HTP compile-time precision, not a quantization pass; a16w8 and
    # a8w8 are 16-/8-bit activations over 8-bit weights, done here before lowering.
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc),
        backend_options=generate_htp_compiler_spec(use_fp16=not quantized),
    )

    if quantized:
        prepared = prepare_pt2e(
            torch_export(model, inputs, strict=False).module(),
            # MinMaxObserver, not the make_quantizer default of
            # MovingAverageMinMaxObserver: the moving average trails the real
            # activation range on this model and costs 1.59 mIoU. Measured on an
            # S26 Ultra over 200 VOC2012 val images: 64.96% with the default,
            # 66.55% with plain min/max, which matches the published XNNPACK
            # int8 file (66.56%) at 3.20 ms/inference. MobileNetV3 a8w8 is the
            # exception, see the module docstring.
            make_quantizer(
                quant_dtype=getattr(QuantDtype, _QUANT_DTYPE[variant["precision"]]),
                soc_model=soc,
                act_observer=HistogramObserver if histogram else MinMaxObserver,
                per_channel_linear=variant["precision"] == "a8w8",
                eps=_eps_for(variant["file"]),
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
