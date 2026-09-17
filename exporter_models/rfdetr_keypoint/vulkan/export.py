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


def _plain_softmax(x, dim, dtype=None):
    """Replaces ``aten._safe_softmax`` with an ordinary softmax.

    Since PyTorch 2.5 the SDPA decomposition emits ``_safe_softmax``, which
    follows the softmax with a guard that zeroes any row whose scores were all
    ``-inf``::

        masked_rows = (scores == -inf).all(dim, keepdim=True)
        out = where(masked_rows, zeros, softmax(scores))

    RF-DETR attends over a fixed-size image with no padding and passes no
    attention mask, so no row can be fully masked and the guard can never fire.
    It is not free: the comparison, the two logical_nots, the reduction, the
    zeros and the select all run over the full attention matrix, which measured
    6.3% of GPU time on this model.

    Only valid for a model that cannot produce an all -inf row. Do not lift this
    into an export that attends over padding.
    """
    return torch.softmax(x, dim=dim, dtype=dtype)


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
        # SDPA is still one node here and the guard only appears once it is
        # decomposed, so the override has to travel with the decomposition that
        # produces it. Taking just the attention entries out of the core table
        # leaves everything else on the path it took before.
        from torch._decomp import core_aten_decompositions

        core = core_aten_decompositions()
        attention_ops = [
            torch.ops.aten.scaled_dot_product_attention.default,
            torch.ops.aten._scaled_dot_product_attention_math.default,
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
        ]
        decompositions = {op: core[op] for op in attention_ops if op in core}
        decompositions[torch.ops.aten._safe_softmax.default] = _plain_softmax
        ep = ep.run_decompositions(decompositions)

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
