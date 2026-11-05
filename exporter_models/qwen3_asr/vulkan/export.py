import importlib
import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# Vulkan's 4-bit groupwise shaders (`linear_dq8ca_q4gsw`, `embedding_q4gsw`)
# are the device-validated path. Decoder linears use group 32: at 64 the first
# generated token loses ~1 nat of confidence on some clips (see NOTES.md).
LINEAR_GROUP_SIZE = 32
EMBEDDING_GROUP_SIZE = 64
# Decoder layers that stay in the graph dtype. Layer 0 does not take 4 bits
# (the model drops its `language <Name><asr_text>` header and derails), and
# Vulkan has no dynamically quantized 8-bit per-channel linear to fall back to.
FP_LAYERS = (0, 1)


def _load_pkg() -> str:
    name = "qwen3_asr_vulkan_pkg"
    if name not in sys.modules:
        pkg = _MODEL_DIR / "_qwen3_asr"
        spec = importlib.util.spec_from_file_location(
            name, pkg / "__init__.py", submodule_search_locations=[str(pkg)]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return name


def _q4(group_size: int):
    from torchao.quantization import MappingType
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig

    return Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(group_size),
        weight_mapping_type=MappingType.SYMMETRIC,
    )


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
        replace_kv_cache_with_custom_kv_cache,
    )
    from executorch.examples.models.llama.source_transformation.quantize import EmbeddingQuantHandler
    from executorch.examples.models.llama.source_transformation.sdpa import replace_sdpa_with_custom_op
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
    from torch.export import Dim
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torchao.quantization import quantize_

    pkg = _load_pkg()
    q = importlib.import_module(pkg)
    ex = importlib.import_module(f"{pkg}.exportable")
    passes = importlib.import_module(f"{pkg}.passes")

    variant = config["variant"]
    max_context = q.MAX_CONTEXT_LEN
    max_prefill = max_context - 1

    m = q.Qwen3Asr()
    params = m.text_params()

    # The STFT front end is a handful of ops on a dynamic-length waveform, run
    # once per call; it stays on XNNPACK. The tower and decoder go to Vulkan.
    ep_pre = ex.export_preprocess(m.preprocessor())
    # `quantize` = "enc:<none|8da4w>,conv:<none|8da8w>" for the audio tower;
    # the decoder recipe is fixed (see NOTES.md).
    recipe = dict(part.split(":") for part in (variant.get("quantize") or "").split(",") if part)
    encoder = m.audio_encoder()
    if recipe.get("enc", "none") == "8da4w":
        # In fp16 the tower's linears are ~80% of its GPU time; the dq8ca/q4gsw
        # kernel runs them on int8 dot products.
        quantize_(encoder, _q4(LINEAR_GROUP_SIZE), filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear))
    # The conv stem runs on XNNPACK either way (see the partitioner below).
    ep_enc = ex.export_audio_encoder(encoder, dynamic=True, quantize_convs=recipe.get("conv", "none") == "8da8w")

    text = m.text_model(
        use_kv_cache=True,
        use_sdpa_with_kv_cache_op=True,
        enable_dynamic_shape=True,
        max_seq_len=max_context,
        max_context_len=max_context,
    )
    text = replace_kv_cache_with_custom_kv_cache(text)
    text = replace_sdpa_with_custom_op(text)
    if variant["quantized"]:
        fp_layers = tuple(f"layers.{i}." for i in FP_LAYERS)
        quantize_(
            text,
            _q4(LINEAR_GROUP_SIZE),
            filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear) and not fqn.startswith(fp_layers),
        )
    decoder = q.TextDecoder(text).eval()
    seq = Dim("seq", min=1, max=max_prefill)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        ep_dec = torch.export.export(
            decoder,
            (torch.randn(1, 8, params["dim"]), torch.tensor([0], dtype=torch.int64)),
            dynamic_shapes=({1: seq}, None),
            strict=True,
        )

    emb = m.token_embedding()
    if variant["quantized"]:
        holder = torch.nn.Module()
        holder.embedding = emb
        emb = EmbeddingQuantHandler(
            holder, bitwidth=4, group_size=EMBEDDING_GROUP_SIZE, packed=True
        ).quantized_model().embedding
    tok = Dim("tokens", min=1, max=max_context)
    with torch.no_grad():
        ep_tok = torch.export.export(
            q.TokenEmbedding(emb),
            (torch.zeros(1, 8, dtype=torch.int64),),
            dynamic_shapes=({1: tok},),
            strict=True,
        )

    schema = ex.model_schema(
        dim=params["dim"], vocab=params["vocab_size"], max_context=max_context,
        max_prefill=max_prefill, dynamic_encoder=True,
    )
    consts = dict(variant.get("constants", {}))
    consts["get_model_schema"] = schema

    def vulkan():
        # fp16 compute, as qwen3/vulkan ships; the seq dim stays symbolic so one
        # method serves both the prefill and the single-token decode.
        return [VulkanPartitioner(compile_options={"require_dynamic_shapes": True, "force_fp16": True})]

    program = to_edge_transform_and_lower(
        {"preprocess": ep_pre, "audio_encoder": ep_enc, "token_embedding": ep_tok, "text_decoder": ep_dec},
        partitioner={
            "preprocess": [XnnpackPartitioner()],
            # Vulkan convolution takes batch 1 only, and the tower convolves its
            # 1 s chunks as a batch; XNNPACK picks up the three conv2d.
            "audio_encoder": vulkan() + [XnnpackPartitioner()],
            "token_embedding": vulkan(),
            "text_decoder": vulkan(),
        },
        transform_passes={
            "preprocess": [],
            "audio_encoder": [],
            "token_embedding": [],
            "text_decoder": [passes.RemoveInversePermutePairs()],
        },
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        constant_methods=consts,
    ).to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
