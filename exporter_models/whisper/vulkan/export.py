"""Whisper on the Vulkan backend.

Only ``encode`` is lowered to Vulkan. ``decode`` stays on XNNPACK, where it
measured faster than the Vulkan equivalent (10.7 ms vs 15.8 ms per step on an
Adreno 840): decode is a single-token step, so it is latency-bound and never
fills the GPU.

Two deliberate departures from the XNNPACK export:

* ``encode`` is exported at a static 30 s of audio. The Vulkan delegate has no
  dynamic-shape support, so the range the other backends accept collapses to
  the one padded length the runtime always feeds it.
* Encoder self-attention is re-expressed over rank-3 tensors. Vulkan packs the
  trailing dimension into texels, so a rank-4 [1, heads, ctx, dim] attention
  wastes three quarters of every texel on the singleton batch. Folding the
  batch into the head dimension is mathematically identical and materially
  faster.

An XNNPACK partitioner is listed after the Vulkan one to catch the mel
preprocessor: its STFT is a conv1d over the full 480000-sample waveform, whose
width-packed texture would need 120100 texels against a 16384 device limit, so
Vulkan cannot take it at any precision.
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

_SAMPLE_RATE = 16000
_AUDIO_SAMPLES = 30 * _SAMPLE_RATE


def _load_custom_whisper():
    pkg_init = _MODEL_DIR / "_custom_whisper" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "whisper_vulkan_custom_whisper", pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_custom_whisper")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["whisper_vulkan_custom_whisper"] = mod
    spec.loader.exec_module(mod)
    return mod


def _qkv_attention_rank3(self, q, k, v, mask=None):
    """Encoder self-attention with the singleton batch folded into the heads."""
    n_batch, n_ctx, n_state = q.shape
    assert n_batch == 1 and mask is None, "encoder-only: batch 1, no causal mask"
    n_head = self.n_head
    head_dim = n_state // n_head
    scale = head_dim**-0.25

    def to_heads(t):
        return t.view(n_ctx, n_head, head_dim).permute(1, 0, 2)

    q3, k3, v3 = to_heads(q), to_heads(k), to_heads(v)
    qk = torch.bmm(q3 * scale, (k3 * scale).transpose(1, 2))
    weights = torch.nn.functional.softmax(qk, dim=-1)
    out = torch.bmm(weights, v3).permute(1, 0, 2).reshape(1, n_ctx, n_state)
    return out, None


class _WhisperPositionalEmbedding(torch.nn.Module):
    def __init__(self, pretrained_weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(pretrained_weight.clone().detach())

    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            return self.weight[
                past_key_values_length : past_key_values_length + input_ids.shape[1]
            ]
        return self.weight[position_ids]


class _EncoderWrapper(torch.nn.Module):
    def __init__(self, preprocessor, encoder):
        super().__init__()
        self._preprocessor = preprocessor
        self._encoder = encoder

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._encoder(self._preprocessor(waveform).unsqueeze(0))


class _DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens, positions, encodings):
        return self.decoder(tokens, positions, encodings)


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from transformers import AutoProcessor

    cw = _load_custom_whisper()
    preprocessor_mod = importlib.import_module(
        "whisper_vulkan_custom_whisper.preprocessor"
    )

    size = config.get("size") or "tiny"
    model = cw.load_model(size)
    model.eval()

    encoder = model.encoder
    encoder.disable_sdpa()
    # Bind per block instance, not on the class: the decoder shares that class
    # and does pass a causal mask, which this implementation asserts against.
    for block in encoder.blocks:
        block.attn.qkv_attention = types.MethodType(_qkv_attention_rank3, block.attn)

    decoder = model.decoder
    if hasattr(decoder, "embed_positions"):
        decoder.embed_positions = _WhisperPositionalEmbedding(
            decoder.embed_positions.weight
        )

    preprocessor = preprocessor_mod.ProcessorTorchModule(
        AutoProcessor.from_pretrained(f"openai/whisper-{size}")
    ).eval()

    encoder_wrapper = _EncoderWrapper(preprocessor, encoder).eval()
    decoder_wrapper = _DecoderWrapper(decoder).eval()

    exported_encode = torch.export.export(
        encoder_wrapper, (torch.randn(_AUDIO_SAMPLES),)
    )
    exported_decode = torch.export.export(
        decoder_wrapper,
        (
            torch.randint(1, 100, size=(1, 1)),
            torch.tensor([0], dtype=torch.long),
            torch.randn(1, 1500, model.dims.n_audio_state),
        ),
    )

    schema_json = load_and_validate_schema(
        _HERE / "schema.json",
        replacements={
            "n_audio_state": str(model.dims.n_audio_state),
            "n_vocab": str(model.dims.n_vocab),
        },
    )

    fp16 = {"force_fp16": True}
    executorch_program = to_edge_transform_and_lower(
        {"encode": exported_encode, "decode": exported_decode},
        partitioner={
            "encode": [VulkanPartitioner(compile_options=fp16), XnnpackPartitioner()],
            "decode": [XnnpackPartitioner()],
        },
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
        transform_passes=[AddmmToLinearTransform()],
        constant_methods={"get_model_schema": schema_json},
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(executorch_program.buffer)
    return str(out_path)
