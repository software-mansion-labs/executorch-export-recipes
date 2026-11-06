"""Whisper -> Qualcomm QNN (HTP) .pte, one per Hexagon version.

Same contract and model adaptations as the Vulkan export (reused from
`../vulkan/export.py`): `encode` takes the raw waveform with a dynamic length
and returns the decoder's cross-attention K/V, hoisted out of decode and packed
head-major; `decode` is one token step against them.

Only `encode` is lowered to QNN, in fp16. The mel preprocessor ahead of it falls
back to XNNPACK, so the dynamic audio length lives only there: everything the
HTP sees is downstream of the fixed [80, 3000] mel and static. `decode` stays on
XNNPACK. On the HTP it runs whole, KV cache included, but at 13 ms per step
against 5.3 ms: a single-token step is latency-bound.

whisper-tiny on an S26 Ultra (v81): encode 36 ms against 166 ms for XNNPACK
(12.5 ms of it is the preprocessor). Over 50 LibriSpeech test-clean clips the
encode output has cosine 0.9999 to fp32 and all 50 transcripts are identical
(WER 7.65% for both). Every quantized encode lost: a16w8 ran at 102 ms
(16-bit attention matmuls), fp16a8w fragmented to 3.2 s, and a8w8 / a16w4
fell to cosine 0.64 / 0.68 on the CPU.

fp16 files on the S26 (median encode; 50 clips, WER fp32 -> device):

    tiny      38 ms   7.65% -> 7.65%
    base      57 ms   4.70% -> 4.70%
    small    148 ms   4.04% -> 4.04%
    tiny.en   39 ms   6.23% -> 6.23%
    base.en   55 ms   4.37% -> 4.37%
    small.en 148 ms   3.93% -> 4.04%

The two variants differ only in the decoder, as in the Vulkan export: `fp16`
keeps it in fp32 and `int8` quantizes its linears for XNNPACK.

Only v81 is device-validated; v69-v79 target SoCs we do not have.
"""

import importlib
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import torch
from torch.export import Dim

from exporter.schema import load_and_validate_schema

_DIR = Path(__file__).resolve().parents[1]

_SAMPLE_RATE = 16000
_AUDIO_SAMPLES = 30 * _SAMPLE_RATE
_MIN_AUDIO_SAMPLES = 201  # one STFT frame; matches the other backends

_HEXAGON_SOC = {
    "v69": "SM8450",  # Snapdragon 8 Gen 1
    "v73": "SM8550",  # Snapdragon 8 Gen 2
    "v75": "SM8650",  # Snapdragon 8 Gen 3
    "v79": "SM8750",  # Snapdragon 8 Elite
    "v81": "SM8850",  # Snapdragon 8 Elite Gen 5
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _soc_for(file_name: str) -> str:
    version = Path(file_name).stem.rsplit("_", 1)[-1]
    if version not in _HEXAGON_SOC:
        raise ValueError(
            f"variant file '{file_name}' must end in a Hexagon version "
            f"{sorted(_HEXAGON_SOC)}; it is what the .pte is compiled for"
        )
    return _HEXAGON_SOC[version]


def export(config: dict, out_dir: str) -> str:
    # Checked before the imports below: without the SDK those fail deep in the
    # QNN AOT chain with an unrelated-looking error.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError(
            "QNN_SDK_ROOT is not set -- the QNN AOT compiler needs the Qualcomm "
            "AI Engine Direct SDK (see skills/backends/qnn)."
        )

    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        QnnManagerContext,
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_qnn_context_binary_alignment,
        get_qnn_pass_manager_cls,
        qnn_edge_config,
    )
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from transformers import AutoProcessor

    variant = config["variant"]
    soc = _soc_for(variant["file"])

    vulkan = _load(_DIR / "vulkan" / "export.py", "whisper_vulkan_for_qnn")
    cw = vulkan._load_custom_whisper()
    preprocessor_mod = importlib.import_module(
        "whisper_vulkan_custom_whisper.preprocessor"
    )

    size = config.get("size") or "tiny"
    model = cw.load_model(size)
    model.eval()

    encoder = model.encoder
    encoder.disable_sdpa()
    for block in encoder.blocks:
        block.attn.qkv_attention = types.MethodType(
            vulkan._qkv_attention_rank3, block.attn
        )

    decoder = model.decoder
    if hasattr(decoder, "embed_positions"):
        decoder.embed_positions = vulkan._WhisperPositionalEmbedding(
            decoder.embed_positions.weight
        )

    preprocessor = preprocessor_mod.ProcessorTorchModule(
        AutoProcessor.from_pretrained(f"openai/whisper-{size}")
    ).eval()

    encoder_wrapper = vulkan._EncoderWrapper(
        preprocessor, encoder, decoder, model.dims.n_text_head
    ).eval()
    projection = vulkan._OutputProjection(
        decoder.token_embedding.weight.detach().clone()
    ).eval()
    if variant["quantized"]:
        from torchao.quantization import quantize_
        from torchao.quantization.granularity import PerAxis
        from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig

        quant_config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int8, weight_granularity=PerAxis(0)
        )
        quantize_(projection, quant_config)

        # cross_attn.key/value run in encode (the hoisted K/V), on the HTP in
        # fp16, so they stay float.
        def _decode_side_linear(module, fqn: str) -> bool:
            if not isinstance(module, torch.nn.Linear):
                return False
            return not fqn.endswith(("cross_attn.key", "cross_attn.value"))

        quantize_(decoder, quant_config, filter_fn=_decode_side_linear)

    decoder_wrapper = vulkan._DecoderWrapper(
        decoder, model.dims.n_text_head, model.dims.n_audio_ctx, projection
    ).eval()

    n_cross_state = model.dims.n_text_layer * 2 * model.dims.n_audio_state

    audio_samples = Dim("T_audio", min=_MIN_AUDIO_SAMPLES, max=_AUDIO_SAMPLES)
    exported_encode = torch.export.export(
        encoder_wrapper,
        (torch.randn(_AUDIO_SAMPLES),),
        dynamic_shapes=({0: audio_samples},),
        strict=False,
    )
    exported_decode = torch.export.export(
        decoder_wrapper,
        (
            torch.randint(1, 100, size=(1, 1)),
            torch.tensor([0], dtype=torch.long),
            torch.randn(1, 1500, n_cross_state),
        ),
        strict=False,
    )

    # Same IO contract as the Vulkan files.
    schema_json = load_and_validate_schema(
        _DIR / "vulkan" / "schema.json",
        replacements={
            "n_cross_state": str(n_cross_state),
            "n_vocab": str(model.dims.n_vocab),
        },
    )

    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc),
        backend_options=generate_htp_compiler_spec(use_fp16=True),
    )
    # What to_edge_transform_and_lower_to_qnn does, for encode only: decode
    # takes no QNN passes, and the preprocessor needs a second partitioner.
    pass_manager = get_qnn_pass_manager_cls()()
    encode = pass_manager.transform_for_export_pipeline(exported_encode)
    with QnnManagerContext({"encode": compiler_specs}):
        edge = to_edge_transform_and_lower(
            {"encode": encode, "decode": exported_decode},
            transform_passes={
                "encode": pass_manager.get_to_edge_transform_passes(
                    exported_encode, compiler_specs=compiler_specs
                ),
                "decode": [],
            },
            partitioner={
                "encode": [QnnPartitioner(compiler_specs), XnnpackPartitioner()],
                "decode": [XnnpackPartitioner()],
            },
            compile_config=qnn_edge_config(),
            constant_methods={"get_model_schema": schema_json},
        )
    program = edge.to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=True, alloc_graph_output=True
            ),
            segment_alignment=get_qnn_context_binary_alignment(),
        )
    )

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    logging.info("wrote %s for %s (%s)", out_path, soc, variant["precision"])
    return str(out_path)
