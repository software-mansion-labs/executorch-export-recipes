"""Shared export path for LFM3-VL (xnnpack / mlx / vulkan).

Three methods, matching RNE's multimodal LLM runner:
  vision_encoder  f32[1, 3, 512, 512] (raw RGB 0..255) -> f32[1, 256, 2048]
  token_embedding i64[1, S]                            -> f32[1, S, 2048]
  text_decoder    f32[1, S, 2048], i64[S] positions    -> f32[1, vocab]
"""

import json
import logging
import os
import shutil
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

SOURCE_REPO = "LiquidAI/LFM3-VL-2B"
SOURCE_REVISION = "f83d2b0a2296aaa277f394990a9b8b949ddf9691"
TEXT_PREFIX = "model.language_model."


def resolve_source() -> Path:
    local = os.environ.get("LFM3_VL_SOURCE")
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


class TokenEmbedding(nn.Module):
    def __init__(self, embed: nn.Embedding):
        super().__init__()
        self.embed = embed

    def forward(self, tokens):
        return self.embed(tokens).float()


class TextDecoder(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, embeds, positions):
        h = self.lm.decode(embeds.to(self.lm.embed_tokens.weight.dtype), positions)
        return self.lm.lm_head(h[:, -1, :]).float()


def model_schema(variant: dict) -> str:
    """Token dims of token_embedding / text_decoder are ranges; the vision
    encoder is fully static."""
    schema = {}
    for name, method in variant["methods"].items():
        sides = {}
        for side in ("inputs", "outputs"):
            sides[side] = []
            for t in method[side]:
                shape = [{"kind": "constant", "value": d} for d in t["shape"]]
                sides[side].append({"kind": "Tensor", "dtype": t["dtype"], "shape": shape})
        if name == "token_embedding":
            seq = method["inputs"][0]["shape"][1]
            sides["inputs"][0]["shape"][1] = _range(seq)
            sides["outputs"][0]["shape"][1] = _range(seq)
        elif name == "text_decoder":
            seq = method["inputs"][0]["shape"][1]
            sides["inputs"][0]["shape"][1] = _range(seq)
            sides["inputs"][1]["shape"][0] = _range(seq)
        schema[name] = {**sides, "runtimeConstraints": []}
    return json.dumps(schema)


def _range(hi: int) -> dict:
    return {"kind": "range", "range": {"min": 1, "max": hi, "step": 1}}


def write_assets(source: Path, out_dir: Path) -> None:
    shutil.copyfile(source / "tokenizer.json", out_dir / "tokenizer.json")
    cfg = json.loads((source / "tokenizer_config.json").read_text())
    template = cfg.get("chat_template") or (source / "chat_template.jinja").read_text()
    # Same monotonic-thinking fix as lfm3_text: never rewrite earlier assistant
    # turns when a user turn is appended.
    needle = "preserve_thinking or loop.index0 > ns.last_user_index"
    if needle not in template:
        raise ValueError("LFM3-VL chat template changed upstream; re-check the thinking fix")
    cfg["chat_template"] = template.replace(needle, "preserve_thinking")
    (out_dir / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    if (source / "LICENSE").exists():
        shutil.copyfile(source / "LICENSE", out_dir / "LICENSE")


def build_all(source: Path, backend: str, ctx: int):
    from .load import build, read_safetensors
    from .vision import VisionConfig, VisionEncoder, convert

    hf_cfg = json.loads((source / "config.json").read_text())
    state = read_safetensors(source)
    vision = VisionEncoder(hf_cfg, (512, 512))
    vstate = {k: v.float() for k, v in convert(state, VisionConfig.from_hf(hf_cfg)).items()}
    missing, unexpected = vision.load_state_dict(vstate, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"vision weight mismatch: {missing} {unexpected}")
    lm = build(
        source, prefix=TEXT_PREFIX, hf_state=state, max_context_len=ctx,
        attn_impl="mlx" if backend == "mlx" else "llama",
        window_slice=backend == "xnnpack",
    )
    return vision.eval(), lm


def export(config: dict, out_dir: str, backend: str) -> str:
    from torch.export import Dim

    variant = config["variant"]
    consts = dict(variant.get("constants", {}))
    methods = variant["methods"]
    seq = methods["text_decoder"]["inputs"][0]["shape"][1]
    hidden = methods["text_decoder"]["inputs"][0]["shape"][2]
    img = methods["vision_encoder"]["inputs"][0]["shape"]
    ctx = consts["get_max_context_len"]

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / variant["file"]
    source = resolve_source()

    if backend == "mlx":
        from executorch.backends.mlx import custom_ops  # noqa: F401
    else:
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

    vision, lm = build_all(source, backend, ctx)
    quantize(vision, lm, backend, variant["precision"])

    eps = {}
    with torch.no_grad():
        eps["vision_encoder"] = torch.export.export(
            vision, (torch.zeros(img),), strict=True
        )
        s = Dim("seq_len", min=1, max=seq)
        eps["token_embedding"] = torch.export.export(
            TokenEmbedding(lm.embed_tokens), (torch.zeros((1, seq), dtype=torch.long),),
            dynamic_shapes=({1: s},), strict=True,
        )
        s2 = Dim("seq_len_dec", min=1, max=seq)
        eps["text_decoder"] = torch.export.export(
            TextDecoder(lm),
            (torch.zeros((1, seq, hidden)), torch.arange(seq, dtype=torch.long)),
            dynamic_shapes=({1: s2}, {0: s2}), strict=True,
        )

    consts["get_model_schema"] = model_schema(variant)
    program = lower(eps, backend, variant, consts)
    out_path.write_bytes(program.buffer)
    logger.info("wrote %s (%.1f MB)", out_path, len(program.buffer) / 2**20)
    write_assets(source, out_dir)
    return str(out_path)


def quantize(vision: nn.Module, lm: nn.Module, backend: str, precision: str) -> None:
    """The vision tower stays unquantized (as in LFM2.5-VL): it is small, and
    its error lands directly on 256 image tokens."""
    if backend == "mlx":
        from executorch.backends.mlx.llm.quantization import quantize_model_

        bits = {"int4": "4w", "int8": "8w"}[precision]
        vision.to(torch.bfloat16)
        lm.to(torch.bfloat16)
        quantize_model_(lm, qlinear_config=bits, qlinear_group_size=64,
                        qembedding_config=bits, qembedding_group_size=64)
        return
    from executorch.extension.llm.export.quantize import quantize_model_

    assert precision == "8da4w", precision
    if backend == "xnnpack":
        quantize_model_(lm, qlinear_config="8da4w", qlinear_group_size=32,
                        qembedding_config="8w", qembedding_group_size=0)
    else:  # vulkan
        quantize_model_(lm, qlinear_config="8da4w", qlinear_group_size=64,
                        qembedding_config="4w", qembedding_group_size=64)


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
            "vision_encoder": [XnnpackPartitioner()],
            "token_embedding": [XnnpackPartitioner()],
            "text_decoder": [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()],
        }
    elif backend == "mlx":
        from executorch.backends.mlx import MLXPartitioner
        from executorch.backends.mlx.passes import get_default_passes
        from exporter.backends import require_mlx_runtime_fork

        require_mlx_runtime_fork()
        passes = get_default_passes()
        parts = {m: [MLXPartitioner()] for m in eps}
    else:
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

        fp16 = {"force_fp16": True}
        parts = {m: [VulkanPartitioner(fp16)] for m in eps}

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
        name = str(node.target).split(":")[0].replace("<EdgeOpOverload", "").strip(" <>")
        if "executorch_call_delegate" in name:
            calls += 1
        elif "getitem" not in name:
            host[name] += 1
    print(f"[delegation] {method}: delegate calls {calls}; host ops {dict(host)}")
    return {"delegate_calls": calls, "host_ops": dict(host)}
