"""torch.export of the backend-independent methods, plus the baked schema."""

import json
from pathlib import Path

import torch
from torch.export import Dim

from .model import (
    AUDIO_TOKENS_PER_CHUNK,
    CHUNK_FRAMES,
    MAX_AUDIO_SAMPLES,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_CHUNKS,
    WINDOW_FRAMES,
)

_SCHEMA = Path(__file__).resolve().parent.parent / "schema.json"

# Shortest waveform the reflect padding accepts (it mirrors 200 samples).
MIN_AUDIO_SAMPLES = 401

# Chat prompt, with the audio embeddings spliced in between the two halves.
PROMPT_PREFIX = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>"
PROMPT_SUFFIX = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"


def export_preprocess(preprocessor: torch.nn.Module) -> torch.export.ExportedProgram:
    samples = Dim("samples", min=MIN_AUDIO_SAMPLES, max=MAX_AUDIO_SAMPLES)
    with torch.no_grad():
        return torch.export.export(
            preprocessor,
            (torch.randn(4 * SAMPLE_RATE),),
            dynamic_shapes=({0: samples},),
            strict=True,
        )


def export_audio_encoder(
    encoder: torch.nn.Module, dynamic: bool, quantize_convs: bool = False
) -> torch.export.ExportedProgram:
    """One attention window. `dynamic` = whole chunks 1..8, else always 8.

    `quantize_convs` runs the conv stem (about half the tower's FLOPs) as
    XNNPACK dynamically quantized int8 conv2d: per-channel weights, per-tensor
    activations chosen at runtime. The tower output moves by ~1% rel L2.
    """
    chunks = WINDOW_CHUNKS if not dynamic else 3
    mel = torch.randn(1, N_MELS, chunks * CHUNK_FRAMES)
    n_valid = torch.tensor([chunks * AUDIO_TOKENS_PER_CHUNK - 4], dtype=torch.int64)
    shapes = None
    if dynamic:
        c = Dim("chunks", min=1, max=WINDOW_CHUNKS)
        shapes = ({2: CHUNK_FRAMES * c}, None)
    with torch.no_grad():
        ep = torch.export.export(encoder, (mel, n_valid), dynamic_shapes=shapes, strict=True)
        if not quantize_convs:
            return ep
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        def conv_only(n: torch.fx.Node) -> bool:
            # The filter also sees each conv's weight/bias get_attr nodes,
            # which carry no module stack, so match on names
            # (set_module_type works once pytorch/executorch#24056 lands).
            if n.op == "get_attr":
                return n.target.startswith("conv2d")
            return n.target == torch.ops.aten.conv2d.default

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        ).set_filter_function(conv_only)
        prepared = prepare_pt2e(ep.module(), quantizer)
        prepared(mel, n_valid)
        return torch.export.export(convert_pt2e(prepared), (mel, n_valid), dynamic_shapes=shapes, strict=True)


def model_schema(*, dim: int, vocab: int, max_context: int, max_prefill: int, dynamic_encoder: bool) -> str:
    from exporter.schema import load_and_validate_schema

    def dim_json(step: int) -> str:
        if not dynamic_encoder:
            return json.dumps({"kind": "constant", "value": WINDOW_CHUNKS * step})
        return json.dumps({"kind": "range", "range": {"min": step, "max": WINDOW_CHUNKS * step, "step": step}})

    return load_and_validate_schema(
        _SCHEMA,
        replacements={
            "mel_frames": dim_json(CHUNK_FRAMES),
            "audio_tokens": dim_json(AUDIO_TOKENS_PER_CHUNK),
            "dim": str(dim),
            "vocab": str(vocab),
            "max_context": str(max_context),
            "max_prefill": str(max_prefill),
        },
    )


def llm_constants(*, tokenizer_eos: list[int], max_context: int, max_prefill: int, n_layers: int, vocab: int) -> dict:
    """Runner-style constant methods, so the .pte also describes itself."""
    return {
        "get_eos_ids": tokenizer_eos,
        "get_max_context_len": max_context,
        "get_max_seq_len": max_prefill,
        "get_n_layers": n_layers,
        "get_vocab_size": vocab,
        "get_audio_window_frames": WINDOW_FRAMES,
        "use_kv_cache": True,
        "enable_dynamic_shape": True,
    }
