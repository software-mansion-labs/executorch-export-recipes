"""Shared export path for LFM3-Audio, audio in / text out (xnnpack / mlx / vulkan).

Three methods, for an audio-input LLM runner:
  audio_encoder   f32[1, 400 + 160k] waveform (16 kHz, k <= max) -> f32[1, T, 2048], T = ceil((k + 1) / 8)
  token_embedding i64[1, S]                                       -> f32[1, S, 2048]
  text_decoder    f32[1, S, 2048], i64[S] positions               -> f32[1, vocab]

Speech output (depth decoder + LiquidCodec) is not exported: the text decoder
masks the tokens that would switch generation to audio.
"""

import json
import logging
import os
import shutil
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

SOURCE_REPO = "LiquidAI/LFM3-Audio-2B"
SOURCE_REVISION = "9d49701c372d6011d900f6abb40c6b296fb706dd"
TEXT_PREFIX = "lfm.model."
HOP, N_FFT = 160, 400
# What upstream generate() suppresses for text-only output (suppress_mixed,
# suppress_audio_out): the <|audio|> / <|audio_out|> placeholders and the
# <|audio_start|> / <|mixed_start|> mode switches.
SUPPRESSED_TOKENS = ["<|audio|>", "<|audio_out|>", "<|audio_start|>", "<|mixed_start|>"]


def resolve_source() -> Path:
    local = os.environ.get("LFM3_AUDIO_SOURCE")
    if local:
        return Path(local).expanduser().resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            SOURCE_REPO,
            revision=SOURCE_REVISION,
            allow_patterns=["*.json", "model.safetensors", "*.jinja", "LICENSE"],
        )
    )


class TokenEmbedding(nn.Module):
    def __init__(self, embed: nn.Embedding):
        super().__init__()
        self.embed = embed

    def forward(self, tokens):
        return self.embed(tokens).float()


class TextDecoder(nn.Module):
    def __init__(self, lm, suppressed_ids):
        super().__init__()
        self.lm = lm
        mask = torch.zeros(lm.cfg.vocab_size)
        mask[suppressed_ids] = float("-inf")
        self.register_buffer("logit_mask", mask, persistent=False)

    def forward(self, embeds, positions):
        h = self.lm.decode(embeds.to(self.lm.embed_tokens.weight.dtype), positions)
        return self.lm.lm_head(h[:, -1, :]).float() + self.logit_mask


def suppressed_ids(source: Path) -> list:
    added = {t["content"]: t["id"] for t in json.loads((source / "tokenizer.json").read_text())["added_tokens"]}
    return [added[t] for t in SUPPRESSED_TOKENS]


def model_schema(variant: dict) -> str:
    """Dynamic dims as RNE ranges: the waveform length steps by the STFT hop
    from one 400-sample window; token dims are 1..seqLen max."""
    schema = {}
    for name, method in variant["methods"].items():
        sides = {}
        for side in ("inputs", "outputs"):
            sides[side] = []
            for t in method[side]:
                shape = [{"kind": "constant", "value": d} for d in t["shape"]]
                sides[side].append({"kind": "Tensor", "dtype": t["dtype"], "shape": shape})
        if name == "audio_encoder":
            samples = method["inputs"][0]["shape"][1]
            tokens = method["outputs"][0]["shape"][1]
            sides["inputs"][0]["shape"][1] = {"kind": "range", "range": {"min": N_FFT, "max": samples, "step": HOP}}
            sides["outputs"][0]["shape"][1] = _range(tokens)
        elif name == "token_embedding":
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
    # The Audio repo ships special_tokens_map.json in place of tokenizer_config.json.
    cfg_path = source / "tokenizer_config.json"
    cfg = json.loads((cfg_path if cfg_path.exists() else source / "special_tokens_map.json").read_text())
    template = cfg.get("chat_template") or (source / "chat_template.jinja").read_text()
    needle = "preserve_thinking or loop.index0 > ns.last_user_index"
    if needle in template:
        # Same monotonic-thinking fix as lfm3_text.
        template = template.replace(needle, "preserve_thinking")
    cfg["chat_template"] = template
    (out_dir / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    if (source / "LICENSE").exists():
        shutil.copyfile(source / "LICENSE", out_dir / "LICENSE")


def build_all(source: Path, backend: str, ctx: int, max_tokens: int):
    from .audio import COMPUTED_BUFFERS, AudioEncoder, EncoderConfig, convert, mel_filters
    from .load import build, read_safetensors

    hf_cfg = json.loads((source / "config.json").read_text())
    state = {k: v for k, v in read_safetensors(source).items()
             if k.startswith(("encoder.", "mlp.", TEXT_PREFIX))}
    audio = AudioEncoder(hf_cfg, mel_filters(), max_tokens=max_tokens)
    astate = {k: v.float() for k, v in convert(state, EncoderConfig.from_hf(hf_cfg)).items()}
    missing, unexpected = audio.load_state_dict(astate, strict=False)
    missing = [k for k in missing if k not in COMPUTED_BUFFERS]
    if missing or unexpected:
        raise RuntimeError(f"audio weight mismatch: {missing} {unexpected}")
    lm = build(
        source, prefix=TEXT_PREFIX, hf_state=state, dtype=None, max_context_len=ctx,
        attn_impl="mlx" if backend == "mlx" else "llama",
        window_slice=backend == "xnnpack",
    )
    return audio.eval(), lm


def export(config: dict, out_dir: str, backend: str) -> str:
    from torch.export import Dim

    variant = config["variant"]
    consts = dict(variant.get("constants", {}))
    methods = variant["methods"]
    seq = methods["text_decoder"]["inputs"][0]["shape"][1]
    hidden = methods["text_decoder"]["inputs"][0]["shape"][2]
    max_samples = methods["audio_encoder"]["inputs"][0]["shape"][1]
    max_tokens = methods["audio_encoder"]["outputs"][0]["shape"][1]
    ctx = consts["get_max_context_len"]
    assert (max_samples - N_FFT) % HOP == 0, max_samples

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / variant["file"]
    source = resolve_source()

    if backend == "mlx":
        from executorch.backends.mlx import custom_ops  # noqa: F401
    else:
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

    audio, lm = build_all(source, backend, ctx, max_tokens)
    quantize(audio, lm, backend, variant["precision"])

    eps = {}
    with torch.no_grad():
        k = Dim("hops", min=0, max=(max_samples - N_FFT) // HOP)
        eps["audio_encoder"] = torch.export.export(
            audio, (torch.zeros((1, max_samples)),),
            dynamic_shapes=({1: HOP * k + N_FFT},), strict=True,
        )
        s = Dim("seq_len", min=1, max=seq)
        eps["token_embedding"] = torch.export.export(
            TokenEmbedding(lm.embed_tokens), (torch.zeros((1, seq), dtype=torch.long),),
            dynamic_shapes=({1: s},), strict=True,
        )
        s2 = Dim("seq_len_dec", min=1, max=seq)
        eps["text_decoder"] = torch.export.export(
            TextDecoder(lm, suppressed_ids(source)),
            (torch.zeros((1, seq, hidden)), torch.arange(seq, dtype=torch.long)),
            dynamic_shapes=({1: s2}, {0: s2}), strict=True,
        )

    consts["get_model_schema"] = model_schema(variant)
    program = lower(eps, backend, variant, consts)
    out_path.write_bytes(program.buffer)
    logger.info("wrote %s (%.1f MB)", out_path, len(program.buffer) / 2**20)
    write_assets(source, out_dir)
    return str(out_path)


def quantize(audio: nn.Module, lm: nn.Module, backend: str, precision: str) -> None:
    """The audio encoder stays unquantized (~100M params), like the VL tower."""
    if backend == "mlx":
        from executorch.backends.mlx.llm.quantization import quantize_model_

        bits = {"int4": "4w", "int8": "8w"}[precision]
        audio.to(torch.bfloat16)
        lm.to(torch.bfloat16)
        quantize_model_(lm, qlinear_config=bits, qlinear_group_size=64,
                        qembedding_config=bits, qembedding_group_size=64)
        return
    assert precision == "8da4w", precision
    if backend == "xnnpack":
        quantize_layerwise(lm, qlinear_config="8da4w", qlinear_group_size=32,
                           qembedding_config="8w", qembedding_group_size=0)
    else:  # vulkan: int4 weights, fp16 compute (force_fp16 in lower())
        quantize_layerwise(lm, qlinear_config="8da4w", qlinear_group_size=64,
                           qembedding_config="4w", qembedding_group_size=64)


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
            "audio_encoder": [XnnpackPartitioner()],
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

        parts = {m: [VulkanPartitioner({"force_fp16": True})] for m in eps}

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
