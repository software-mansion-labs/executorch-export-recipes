"""ZipDepth base -> Qualcomm QNN (HTP) .pte, one per Hexagon version.

Same contract as the other backends: float32 RGB [1, 3, 384, 384] in [0, 1]
(ImageNet normalization is inside the model), raw depth [1, 384, 384]. fp16 on
the HTP, one QNN partition, the Hexagon version read from the file suffix.

As exported for the other backends the HTP output is wrong (NYUv2 AbsRel 18.2%
against fp32 13.6%, worst-image correlation with fp32 0.26). Two changes, both
exact in fp32 (1.5e-6 relative), fix it:

* Every BatchNorm is folded into the conv(s) feeding it. Ten stay unfused
  upstream, five after a sum of two convs, and QNN runs them as a standalone
  Batchnorm op that loses accuracy on the HTP (it also ignored the op's eps:
  pytorch/executorch#24501).
* GlobalContextBlock pools with mul + sum instead of
  `bmm(x.view(B, C, HW), mask.transpose(1, 2)).unsqueeze(-1)`, which the HTP
  computes as roughly 1/24 of the right value.

On an S26 Ultra, over the 145 NYUv2 test images (affine-aligned against GT):
fp16 AbsRel 13.56% / delta1 83.54% against fp32 13.55% / 83.57%, at 3.0 ms per
inference (shipped Vulkan fp16: 11.5 ms). a8w8 with the same changes runs at
0.82 ms with AbsRel 13.97%, but five images fall below 0.95 correlation with
fp32 (the worst 0.53) while CPU fake-quant keeps them above 0.9, so it is not
shipped. Only v81 is device-validated.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_DIR = Path(__file__).resolve().parents[1]

_HEXAGON_SOC = {
    "v69": "SM8450",  # Snapdragon 8 Gen 1
    "v73": "SM8550",  # Snapdragon 8 Gen 2
    "v75": "SM8650",  # Snapdragon 8 Gen 3
    "v79": "SM8750",  # Snapdragon 8 Elite
    "v81": "SM8850",  # Snapdragon 8 Elite Gen 5
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fold_batch_norms(model: torch.nn.Module) -> None:
    """Fold each BatchNorm into the conv, or the two summed convs, before it."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    for module in list(model.modules()):
        bn = getattr(module, "bn", None)
        if not isinstance(bn, torch.nn.BatchNorm2d):
            continue
        if isinstance(getattr(module, "conv", None), torch.nn.Conv2d):
            module.conv = fuse_conv_bn_eval(module.conv, bn)
        else:
            # MinimalMultiScale / UltraLightFusion: bn(conv_a(x) + conv_b(y)).
            pair = [
                getattr(module, name)
                for name in ("branch1", "branch2", "proj_high", "proj_low")
                if hasattr(module, name)
            ]
            assert len(pair) == 2, f"unexpected BatchNorm site {type(module).__name__}"
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            shift = bn.bias - bn.running_mean * scale
            with torch.no_grad():
                for index, conv in enumerate(pair):
                    assert conv.bias is None
                    conv.weight.mul_(scale.view(-1, 1, 1, 1))
                    conv.bias = torch.nn.Parameter(
                        shift.clone() if index == 0 else torch.zeros_like(shift)
                    )
        module.bn = torch.nn.Identity()


def _global_context_sum(self, x):
    """GlobalContextBlock with its softmax-weighted pooling as mul + sum."""
    batch, channels, height, width = x.shape
    mask = F.softmax(self.context_weight(x).view(batch, 1, height * width), dim=-1)
    pooled = x.view(batch, channels, height * width) * mask
    context = pooled.sum(-1, keepdim=True).unsqueeze(-1)
    return x + self.transform(context)


def _build_model():
    model = _load(_DIR / "_model.py", "zipdepth_model_qnn").build_model()
    vulkan = _load(_DIR / "vulkan" / "export.py", "zipdepth_vulkan_for_qnn")
    from zipdepth.model.architecture import GlobalContextBlock, MinimalCrossScale

    for layer in model.modules():
        if isinstance(layer, MinimalCrossScale):
            layer.forward = vulkan._cross_scale_regular_pool.__get__(layer)
        if isinstance(layer, GlobalContextBlock):
            layer.forward = _global_context_sum.__get__(layer)
    _fold_batch_norms(model)
    return model.eval()


def _soc_for(file_name: str) -> str:
    version = Path(file_name).stem.rsplit("_", 1)[-1]
    if version not in _HEXAGON_SOC:
        raise ValueError(
            f"variant file '{file_name}' must end in a Hexagon version "
            f"{sorted(_HEXAGON_SOC)}; it is what the .pte is compiled for"
        )
    return _HEXAGON_SOC[version]


def export(config: dict, out_dir: str) -> str:
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

    model = _build_model()
    inputs = (torch.zeros(shape, dtype=torch.float32),)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc),
        backend_options=generate_htp_compiler_spec(use_fp16=True),
    )
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
    logging.info("wrote %s for %s (%s)", out_path, soc, variant["precision"])
    return str(out_path)
