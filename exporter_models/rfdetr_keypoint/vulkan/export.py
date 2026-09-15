import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_model():
    name = "rfdetr_kp_model_vulkan"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_keypoint_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _detach_propagated_constants(ep):
    """Detaches the tensors `constant_prop_pass` leaves attached to autograd.

    A constant shared by two partitions is deepcopied into each, and deepcopy
    refuses a non-leaf tensor, so the lowering dies well after the pass that
    caused it.
    """
    for key, value in list(ep.constants.items()):
        if isinstance(value, torch.Tensor) and (
            value.requires_grad or value.grad_fn is not None
        ):
            ep.constants[key] = value.detach().clone()
    for key, value in list(ep.state_dict.items()):
        if isinstance(value, torch.nn.Parameter) and value.grad_fn is not None:
            ep.state_dict[key] = torch.nn.Parameter(
                value.detach().clone(), requires_grad=False
            )
    return ep


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.passes.constant_prop_pass import constant_prop_pass

    wrapper, example = _load_model().build_wrapper()

    # no_grad, not inference_mode: inference-mode tensors are rejected during
    # Vulkan lowering even though XNNPACK accepts them.
    with torch.no_grad():
        ep = torch_export(wrapper, args=example, strict=False)
        # Re-trace so every fake tensor shares one FakeTensorMode. strict=False
        # leaves the batch norm parameters under a second mode, and Vulkan's
        # FuseBatchNormPass then trips detect_fake_mode.
        ep = ep.run_decompositions({})

    # Fold the arange-built positional grids. Without this the keypoint export
    # does not lower at all, and they would be recomputed every inference.
    ep = _detach_propagated_constants(constant_prop_pass(ep))

    program = to_edge_transform_and_lower(
        ep,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=[VulkanPartitioner(compile_options={"force_fp16": True})],
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
