import importlib.util
import sys
from pathlib import Path

import torch


def _load_model():
    name = "depth_anything_v3_model_vulkan"
    path = Path(__file__).resolve().parents[1] / "_model.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_model()


def _static_rope(self, tokens, positions):
    """Use the known patch-grid extent instead of extracting a tensor scalar."""
    feature_dim = tokens.size(-1) // 2
    cos, sin = self._compute_frequency_components(
        feature_dim, self._export_max_position, tokens.device, tokens.dtype
    )
    vertical, horizontal = tokens.chunk(2, dim=-1)
    return torch.cat(
        (
            self._apply_1d_rope(vertical, positions[..., 0], cos, sin),
            self._apply_1d_rope(horizontal, positions[..., 1], cos, sin),
        ),
        dim=-1,
    )


def _fold_constants(graph):
    """Fold image-independent positional tensors after functionalization."""
    from torch.fx.node import map_arg

    values = {}
    for node in list(graph.graph.nodes):
        if node.op == "get_attr":
            value = graph
            for component in node.target.split("."):
                value = getattr(value, component)
            values[node] = value
        elif node.op == "call_function" and all(
            n in values for n in node.all_input_nodes
        ):
            args = map_arg(node.args, lambda n: values[n])
            kwargs = map_arg(node.kwargs, lambda n: values[n])
            value = node.target(*args, **kwargs)
            values[node] = value
            # Keep large spatial embeddings as broadcast expressions.
            if isinstance(value, torch.Tensor) and (
                value.numel() <= 100_000 or value.dtype in (torch.int8, torch.uint8)
            ):
                name = f"_export_constant_{len(values)}"
                graph.register_buffer(
                    name, value.detach().clone(memory_format=torch.contiguous_format)
                )
                with graph.graph.inserting_before(node):
                    replacement = graph.graph.get_attr(name)
                    replacement.meta = {
                        k: v
                        for k, v in node.meta.items()
                        if k not in ("val", "tensor_meta")
                    }
                node.replace_all_uses_with(replacement)
                values[replacement] = value
    graph.graph.eliminate_dead_code()
    graph.graph.lint()
    graph.recompile()


def verify_delegate_arity(buffer):
    """Check serialized Vulkan IO against every runtime delegate invocation."""
    from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
        VulkanDelegateHeader,
        flatbuffer_to_vk_graph,
    )
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import DelegateCall

    program = deserialize_pte_binary(buffer).program
    checks = []
    for plan in program.execution_plan:
        for chain in plan.chains:
            for instruction in chain.instructions:
                call = instruction.instr_args
                if not isinstance(call, DelegateCall):
                    continue
                delegate = plan.delegates[call.delegate_index]
                if delegate.id != "VulkanBackend":
                    continue
                payload = bytes(
                    program.backend_delegate_data[delegate.processed.index].data
                )
                header = VulkanDelegateHeader.from_bytes(payload[:30])
                graph = flatbuffer_to_vk_graph(
                    payload[
                        header.flatbuffer_offset : header.flatbuffer_offset
                        + header.flatbuffer_size
                    ]
                )
                expected = len(graph.input_ids) + len(graph.output_ids)
                assert len(call.args) == expected, (
                    f"Vulkan delegate {call.delegate_index}: {len(call.args)} arguments, "
                    f"{len(graph.input_ids)} inputs + {len(graph.output_ids)} outputs"
                )
                checks.append(
                    {
                        "arguments": len(call.args),
                        "inputs": len(graph.input_ids),
                        "outputs": len(graph.output_ids),
                    }
                )
    assert checks, "No Vulkan delegate invocation found"
    return checks


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    model = _load_model()
    from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D

    for layer in model.modules():
        if hasattr(layer, "fused_attn"):
            layer.fused_attn = False
        if isinstance(layer, RotaryPositionEmbedding2D):
            layer._export_max_position = max(shape[-2:]) // 14 + 1
            layer.forward = _static_rope.__get__(layer)
    torch.set_num_threads(4)
    inputs = (torch.zeros(shape, dtype=torch.float32),)
    with torch.no_grad():
        model(*inputs)
        # Functionalize before promoting computed constants to buffers. Upstream
        # positional embeddings divide a freshly created tensor in-place; folding
        # its producer first turns that into a persistent buffer mutation, which
        # Vulkan incorrectly serializes as an extra delegate output.
        graph = (
            torch.export.export(model, inputs, strict=False)
            .run_decompositions({})
            .module()
        )
        _fold_constants(graph)
        ep = torch.export.export(graph, inputs, strict=False)
        assert not ep.graph_signature.buffers_to_mutate, (
            "Constant folding introduced buffer mutations"
        )
    program = to_edge_transform_and_lower(
        ep,
        partitioner=[
            VulkanPartitioner(
                compile_options={
                    "force_fp16": True,
                    "texture_limits": (2048, 2048, 2048),
                }
            )
        ],
    ).to_executorch()

    path = Path(out_dir) / variant["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = program.buffer
    verify_delegate_arity(buffer)
    path.write_bytes(buffer)
    return str(path)
