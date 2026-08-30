"""Whisper on the Vulkan backend.

Only ``encode`` is lowered to Vulkan. ``decode`` stays on XNNPACK, where it
measured faster than the Vulkan equivalent (10.7 ms vs 15.8 ms per step on an
Adreno 840): decode is a single-token step, so it is latency-bound and never
fills the GPU.

Two deliberate departures from the XNNPACK export:

* ``encode`` keeps the dynamic audio length the other backends accept, even
  though the Vulkan delegate has no dynamic-shape support. It does not need it:
  the mel preprocessor pads and slices to a fixed ``[80, 3000]``, so every
  tensor Vulkan sees downstream of that is static, and the symbolic length
  lives only in the XNNPACK-delegated STFT ahead of it. Shorter audio is
  correspondingly cheaper (54 ms at 201 samples against 62 ms at 30 s) rather
  than being padded up to full cost.
* Encoder self-attention is re-expressed over rank-3 tensors. Vulkan packs the
  trailing dimension into texels, so a rank-4 [1, heads, ctx, dim] attention
  wastes three quarters of every texel on the singleton batch. Folding the
  batch into the head dimension is mathematically identical and materially
  faster.

``encode`` also returns the decoder's cross-attention K/V, packed as
``[1, n_audio_ctx, n_layer * 2 * n_state]``, instead of the raw encoder states.
Whisper's cross-attention keys and values depend only on the audio, but the
model recomputed them from scratch on every decode step, over all 1500 encoder
frames in every block. Hoisting them into ``encode`` costs 6.5 ms there and
saves 4.9 ms per decode step, so it breaks even after 1.34 steps and wins on
every real transcription. Output is bit-identical.

They are emitted head-major, that is ``[n_head, n_audio_ctx, head_dim]``
flattened back into the transport shape, rather than in the natural frame-major
order. Attention wants the head dimension outermost, and the usual
``view().permute()`` to get there is a full transpose of 4.6 MB per block. Doing
it frame-major meant paying that on 18.4 MB on every decode step; doing it here
pays it once. Worth a further 1.78 ms per step against 3.44 ms in encode, so it
breaks even after 1.93 steps. Also bit-identical.

Two variants are exported. The default keeps the output projection in fp32.
The quantized one takes it to int8, which is the single largest item left in a
decode step: the projection streams the whole 384x51865 tied embedding, 79.7 MB
of fp32, on every token, and at 43 GB/s that is genuinely bandwidth-bound, so
only fewer bytes helps. It is worth 1.46 ms per step and 59 MB of file, and it
costs accuracy: on 8 librispeech clips it moved WER from 10.50% to 12.00% and
changed 4 of 8 transcripts, mostly punctuation but not only. int4 was measured
too and rejected: 0.18 ms faster than int8, but it stopped emitting
end-of-transcript and looped.

Only the projection is quantized, not the whole model. The encoder's weights
have to stay in a form the Vulkan partitioner will take.

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
from torch.export import Dim

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

_SAMPLE_RATE = 16000
_AUDIO_SAMPLES = 30 * _SAMPLE_RATE
_MIN_AUDIO_SAMPLES = 201  # one STFT frame; matches the other backends


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
    """Mel preprocessor + encoder, emitting the decoder's cross-attention K/V.

    The cross-attention keys and values are a function of the audio alone, so
    they are computed once here rather than on every decode step. They are
    packed along the last dimension as ``[k_0, v_0, k_1, v_1, ...]``, one pair
    per decoder block, each already laid out head-major so that
    ``_DecoderWrapper`` can read it as ``[n_head, n_audio_ctx, head_dim]``
    without transposing.
    """

    def __init__(self, preprocessor, encoder, decoder, n_head: int):
        super().__init__()
        self._preprocessor = preprocessor
        self._encoder = encoder
        self._decoder = decoder
        self._n_head = n_head

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        states = self._encoder(self._preprocessor(waveform).unsqueeze(0))
        _, n_ctx, n_state = states.shape
        head_dim = n_state // self._n_head

        cross_kv = []
        for block in self._decoder.blocks:
            for projection in (block.cross_attn.key, block.cross_attn.value):
                projected = projection(states)
                cross_kv.append(
                    projected.view(n_ctx, self._n_head, head_dim)
                    .permute(1, 0, 2)
                    .reshape(1, n_ctx, n_state)
                )
        return torch.cat(cross_kv, dim=-1)


class _OutputProjection(torch.nn.Module):
    """The tied-embedding output projection, as a Linear so it can be quantized.

    Kept separate from ``token_embedding`` deliberately: the input lookup reads
    one row per step and is free, while this streams the entire matrix, so only
    this copy is worth quantizing.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.proj = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(weight)

    def forward(self, x):
        return self.proj(x)


class _DecoderWrapper(torch.nn.Module):
    """One decode step, taking the cross-attention K/V ``_EncoderWrapper`` emitted.

    This mirrors ``TextDecoder.forward`` exactly apart from the cross-attention
    inputs, which are sliced out of ``cross_kv`` rather than projected from the
    encoder states. Because those slices arrive head-major, the attention here
    is plain ``bmm`` over rank-3 tensors and never transposes them; only the
    query, 384 elements, is reshaped. Self-attention keeps its own KV cache and
    is untouched.
    """

    def __init__(self, decoder, n_head: int, n_audio_ctx: int, projection):
        super().__init__()
        self.decoder = decoder
        self._n_head = n_head
        self._n_audio_ctx = n_audio_ctx
        self.projection = projection

    def forward(self, tokens, positions, cross_kv):
        decoder = self.decoder
        n_head, n_ctx = self._n_head, self._n_audio_ctx
        n_state = decoder.token_embedding.embedding_dim
        head_dim = n_state // n_head
        scale = head_dim**-0.25

        x = decoder.token_embedding(tokens) + decoder.positional_embedding[positions]
        x = x.to(cross_kv.dtype)

        for i, block in enumerate(decoder.blocks):
            x = x + block.attn(block.attn_ln(x), positions, mask=decoder.mask)[0]
            cross_attn = block.cross_attn

            key = cross_kv[..., (2 * i) * n_state : (2 * i + 1) * n_state]
            value = cross_kv[..., (2 * i + 1) * n_state : (2 * i + 2) * n_state]
            key = key.reshape(n_head, n_ctx, head_dim)
            value = value.reshape(n_head, n_ctx, head_dim)

            query = cross_attn.query(block.cross_attn_ln(x))
            query = query.view(1, n_head, head_dim).permute(1, 0, 2)
            scores = torch.bmm(query * scale, (key * scale).transpose(1, 2))
            weights = torch.nn.functional.softmax(scores, dim=-1)
            attended = (
                torch.bmm(weights, value).permute(1, 0, 2).reshape(1, 1, n_state)
            )

            x = x + cross_attn.out(attended)
            x = x + block.mlp(block.mlp_ln(x))

        return self.projection(decoder.ln(x)).float()


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

    encoder_wrapper = _EncoderWrapper(
        preprocessor, encoder, decoder, model.dims.n_text_head
    ).eval()
    projection = _OutputProjection(decoder.token_embedding.weight.detach().clone()).eval()
    if config["variant"]["quantized"]:
        from torchao.quantization import quantize_
        from torchao.quantization.granularity import PerAxis
        from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig

        quantize_(
            projection,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int8, weight_granularity=PerAxis(0)
            ),
        )

    decoder_wrapper = _DecoderWrapper(
        decoder, model.dims.n_text_head, model.dims.n_audio_ctx, projection
    ).eval()

    n_cross_state = model.dims.n_text_layer * 2 * model.dims.n_audio_state

    # Matches the range the other whisper backends declare. The Vulkan blobs sit
    # behind the preprocessor's fixed-size mel, so they never see this symbol.
    audio_samples = Dim("T_audio", min=_MIN_AUDIO_SAMPLES, max=_AUDIO_SAMPLES)
    exported_encode = torch.export.export(
        encoder_wrapper,
        (torch.randn(_AUDIO_SAMPLES),),
        dynamic_shapes=({0: audio_samples},),
    )
    exported_decode = torch.export.export(
        decoder_wrapper,
        (
            torch.randint(1, 100, size=(1, 1)),
            torch.tensor([0], dtype=torch.long),
            torch.randn(1, 1500, n_cross_state),
        ),
    )

    schema_json = load_and_validate_schema(
        _HERE / "schema.json",
        replacements={
            "n_cross_state": str(n_cross_state),
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
