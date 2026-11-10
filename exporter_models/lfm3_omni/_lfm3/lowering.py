"""Model-owned lowering and quantization helpers, adapted from LFM3 recipes."""
import torch
def quantize_layerwise(model: torch.nn.Module, **kw) -> None:
    """`quantize_model_` one decoder layer at a time, each upcast to fp32 just
    before it is quantized, so the export never holds a full fp32 copy of the
    model (~7 GB at 1.7B). Per-module quantization makes the result identical
    to quantizing the whole fp32 model at once."""
    from executorch.extension.llm.export.quantize import quantize_model_

    linear_kw = {k: v for k, v in kw.items() if k.startswith("qlinear")}
    for layer in model.layers:
        layer.float()
        quantize_model_(layer, **linear_kw)
    head = torch.nn.Module()
    head.embed_tokens, head.lm_head = model.embed_tokens, model.lm_head
    head.float()
    quantize_model_(head, **kw)
    model.float()


def lower(eps: dict, backend: str, variant: dict, consts: dict):
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

    passes = []
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner, XnnpackPartitioner,
        )

        parts = {
            **{name: [XnnpackPartitioner()] for name in eps},
            "text_decoder": [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()],
        }
    elif backend == "mlx":
        from executorch.backends.mlx import MLXPartitioner
        from executorch.backends.mlx.passes import get_default_passes
        from exporter.backends import require_mlx_runtime_fork

        require_mlx_runtime_fork()
        passes = get_default_passes()
        parts = {m: [MLXPartitioner()] for m in eps}

    edge = to_edge_transform_and_lower(
        eps,
        transform_passes=passes,
        partitioner=parts,
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        constant_methods=consts,
    )
    for name in eps:
        report_delegation(name, edge.exported_program(name))
    return edge.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )


def report_delegation(method: str, ep) -> dict:
    from collections import Counter

    calls, host = 0, Counter()
    for node in ep.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", str(node.target))
        if "executorch_call_delegate" in name:
            calls += 1
        elif "getitem" not in name:
            host[name] += 1
    print(f"[delegation] {method}: delegate calls {calls}; host ops {dict(host)}")
    return {"delegate_calls": calls, "host_ops": dict(host)}
