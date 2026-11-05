import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_model():
    name = "rfdetr_kp_model_xnnpack"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_keypoint_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_FP32_LINEARS = ("sampling_offsets", "attention_weights", "class_embed", "bbox_embed", "keypoint")


def _quantize_linears(module: nn.Module, scheme: str):
    """Dynamic int8 activations + int8 per-channel weights on Linears (convs stay fp32)."""
    from torchao.quantization.granularity import PerAxis
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_

    assert scheme == "8da8w-pc", scheme
    qcfg = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int8, weight_granularity=PerAxis(0)
    )
    # exact type: MultiheadAttention's out_proj subclass is used through its raw weight.
    # Deformable-attention offsets/weights and the class/box/keypoint heads stay fp32:
    # all-int8 costs 0.73 OKS mAP in an eager screen, this split 0.08.
    quantize_(
        module,
        qcfg,
        filter_fn=lambda mod, fqn: type(mod) is nn.Linear
        and not any(s in fqn for s in _FP32_LINEARS),
    )


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    wrapper, example = _load_model().build_wrapper()
    variant = config["variant"]
    if variant.get("quantize"):
        _quantize_linears(wrapper, variant["quantize"])
    # torchao's quantized weights cannot be traced under inference_mode
    grad_ctx = torch.no_grad if variant.get("quantize") else torch.inference_mode
    with grad_ctx():
        ep = torch_export(wrapper, args=example, strict=False)

    program = to_edge_transform_and_lower(
        ep,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
