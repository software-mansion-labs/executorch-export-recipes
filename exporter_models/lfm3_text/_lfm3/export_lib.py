"""Shared export path for the LFM3 text decoder (xnnpack / mlx / vulkan)."""

import json
import logging
import os
import shutil
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

SOURCE_REPO = "LiquidAI/LFM3-1.7B"
SOURCE_REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"


def resolve_source() -> Path:
    """Local snapshot of the pinned upstream checkpoint. `LFM3_SOURCE` points
    at an existing snapshot directory instead of downloading."""
    local = os.environ.get("LFM3_SOURCE")
    if local:
        return Path(local).expanduser().resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            SOURCE_REPO,
            revision=SOURCE_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "LICENSE"],
        )
    )


def model_schema(variant: dict) -> str:
    """`get_model_schema` for RNE: the forward token dim is a range, so the
    runner binds its prefill chunk to the range max."""
    schema = {}
    for name, method in variant["methods"].items():
        sides = {}
        for side in ("inputs", "outputs"):
            sides[side] = []
            for i, t in enumerate(method[side]):
                shape = [{"kind": "constant", "value": d} for d in t["shape"]]
                if name == "forward" and side == "inputs" and i == 0:
                    shape[1] = {"kind": "range", "range": {"min": 1, "max": t["shape"][1], "step": 1}}
                sides[side].append({"kind": "Tensor", "dtype": t["dtype"], "shape": shape})
        schema[name] = {**sides, "runtimeConstraints": []}
    return json.dumps(schema)


def write_assets(source: Path, out_dir: Path) -> None:
    """tokenizer.json + tokenizer_config.json (chat template embedded) + LICENSE."""
    shutil.copyfile(source / "tokenizer.json", out_dir / "tokenizer.json")
    cfg = json.loads((source / "tokenizer_config.json").read_text())
    template = cfg.get("chat_template") or (source / "chat_template.jinja").read_text()
    # Upstream keeps the thinking of assistant turns after the last user
    # message, so appending a user turn rewrites earlier output and breaks the
    # RNE chat preprocessor's prefix check. Strip it from every assistant turn
    # (same fix as the LFM2.5 2.6B and Qwen3 templates); the generation prompt
    # still opens <think>.
    needle = "preserve_thinking or loop.index0 > ns.last_user_index"
    if needle not in template:
        raise ValueError("LFM3 chat template changed upstream; re-check the thinking fix")
    cfg["chat_template"] = template.replace(needle, "preserve_thinking")
    (out_dir / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    if (source / "LICENSE").exists():
        shutil.copyfile(source / "LICENSE", out_dir / "LICENSE")


def export(config: dict, out_dir: str, backend: str) -> str:
    from torch.export import Dim

    from .load import build

    variant = config["variant"]
    consts = dict(variant.get("constants", {}))
    fwd = variant["methods"]["forward"]
    max_seq = fwd["inputs"][0]["shape"][1]
    ctx = consts["get_max_context_len"]

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / variant["file"]
    source = resolve_source()

    if backend == "mlx":
        from executorch.backends.mlx import custom_ops  # noqa: F401  registers mlx::*
    else:
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401  llama::*

    model = build(
        source,
        dtype=None,
        max_context_len=ctx,
        attn_impl="mlx" if backend == "mlx" else "llama",
        window_slice=backend == "xnnpack",
    )
    quantize(model, backend, variant["precision"])

    tokens = torch.zeros((1, max_seq), dtype=torch.long)
    input_pos = torch.tensor([0], dtype=torch.long)
    dyn = {"tokens": {1: Dim("seq_len", min=1, max=max_seq)}, "input_pos": None}
    logger.info("torch.export (seq_len 1..%d, context %d)", max_seq, ctx)
    with torch.no_grad():
        ep = torch.export.export(model, (tokens, input_pos), dynamic_shapes=dyn, strict=True)

    consts["get_model_schema"] = model_schema(variant)
    program = lower(ep, backend, variant, consts)
    out_path.write_bytes(program.buffer)
    logger.info("wrote %s (%.1f MB)", out_path, len(program.buffer) / 2**20)
    write_assets(source, out_dir)
    return str(out_path)


def quantize(model: torch.nn.Module, backend: str, precision: str) -> None:
    if backend == "xnnpack":
        assert precision == "8da4w", precision
        # Same recipe as the LFM2.5 2.6B XNNPACK export: 8da4w gs32 linears
        # (lm_head included) + per-row int8 embedding table.
        quantize_layerwise(
            model, qlinear_config="8da4w", qlinear_group_size=32,
            qembedding_config="8w", qembedding_group_size=0,
        )
    elif backend == "mlx":
        from executorch.backends.mlx.llm.quantization import quantize_model_

        bits = {"int4": "4w", "int8": "8w"}[precision]
        model.to(torch.bfloat16)
        quantize_model_(
            model, qlinear_config=bits, qlinear_group_size=64,
            qembedding_config=bits, qembedding_group_size=64,
        )
    elif backend == "vulkan":
        assert precision == "8da4w", precision
        # int4 weights, fp16 compute (force_fp16 in lower()), as LFM2.5 2.6B.
        quantize_layerwise(
            model, qlinear_config="8da4w", qlinear_group_size=64,
            qembedding_config="4w", qembedding_group_size=64,
        )
    else:
        raise ValueError(backend)


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


def lower(ep, backend: str, variant: dict, consts: dict):
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

    passes = []
    compile_config = EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner, XnnpackPartitioner,
        )

        partitioners = [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]
    elif backend == "mlx":
        from executorch.backends.mlx import MLXPartitioner
        from executorch.backends.mlx.passes import get_default_passes
        from exporter.backends import require_mlx_runtime_fork

        require_mlx_runtime_fork()
        passes = get_default_passes()
        partitioners = [MLXPartitioner()]
    else:
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

        partitioners = [VulkanPartitioner({"force_fp16": variant["precision"] != "fp32"})]

    edge = to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=passes,
        partitioner=partitioners,
        compile_config=compile_config,
        constant_methods=consts,
    )
    report_delegation(edge.exported_program("forward"))
    return edge.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            # Fuses torchao dequantize_affine + embedding into
            # quantized_decomposed::embedding_byte (as LLMEdgeManager does).
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )


def report_delegation(ep) -> dict:
    """Count delegate calls and the ops left on the portable CPU kernels."""
    from collections import Counter

    calls, host = 0, Counter()
    for node in ep.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        name = str(node.target)
        if "executorch_call_delegate" in name:
            calls += 1
        elif name not in ("<built-in function getitem>",):
            host[name] += 1
    logger.info("delegate calls: %d; host ops: %s", calls, dict(host))
    print(f"[delegation] delegate calls: {calls}; host ops: {dict(host)}")
    return {"delegate_calls": calls, "host_ops": dict(host)}
