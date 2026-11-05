"""Opus-MT on Vulkan: encode on the GPU in fp16, decode on XNNPACK.

``decode`` is one token of work per call, latency bound, and ends in a
512 x vocab projection whose output row is wider than a Vulkan texture, so it
stays on XNNPACK with the same int8 recipe as the XNNPACK variant (8da8w
per-channel Linears, int8 embedding table, decomposed layer norms).

``encode`` keeps the dynamic source length; the encoder and the cross-K/V
projections run in one Vulkan delegate.
"""

import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256


def _load(name: str):
    key = f"opus_mt_vulkan{name}"
    spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower

    marian = _load("_marian")
    variant = config["variant"]
    model, cfg = marian.load(marian.source_repo(config["size"]), MAX_TARGET_LEN)

    # Decode-side weights take the XNNPACK int8 recipe. The encoder (and the
    # cross K/V projections it runs) stays fp32 for the Vulkan partition.
    def decode_side(module, fqn):
        return isinstance(module, torch.nn.Linear) and not fqn.startswith("encoder_layers.") \
            and not fqn.endswith(("encoder_attn.k_proj", "encoder_attn.v_proj"))

    from torchao.quantization.granularity import PerAxis
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_

    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8, weight_granularity=PerAxis(0)),
        filter_fn=decode_side,
    )

    # XNNPACK has no layer_norm: decompose the decoder's norms so decode stays
    # one delegate per block. The encoder keeps Vulkan's native layer_norm.
    for layer in model.decoder_layers:
        for norm in (layer.self_attn_layer_norm, layer.encoder_attn_layer_norm, layer.final_layer_norm):
            norm.decomposed = True

    from executorch.examples.models.llama.source_transformation.quantize import (
        EmbeddingQuantHandler,
    )

    model = EmbeddingQuantHandler(
        model, bitwidth=8, group_size=None, packed=False, precision=torch.float32
    ).quantized_model()

    methods, schema = marian.export_methods(model, cfg, MAX_SOURCE_LEN, static=False)
    program = to_edge_transform_and_lower(
        methods,
        partitioner={
            "encode": [VulkanPartitioner(compile_options={"force_fp16": True}), XnnpackPartitioner()],
            "decode": [XnnpackPartitioner()],
        },
        compile_config=EdgeCompileConfig(_core_aten_ops_exception_list=[torch.ops.aten.linear.default]),
        transform_passes=[AddmmToLinearTransform()],
        constant_methods={"get_model_schema": marian.schema_json(schema)},
    ).to_executorch(ExecutorchBackendConfig(run_reinplace_pass=True))

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    _load("_tokenizer").write(marian.source_repo(config["size"]), out_path.parent / "tokenizer.json")
    return str(out_path)
