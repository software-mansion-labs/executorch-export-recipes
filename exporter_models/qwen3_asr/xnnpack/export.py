import importlib
import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_pkg() -> str:
    name = "qwen3_asr_xnnpack_pkg"
    if name not in sys.modules:
        pkg = _MODEL_DIR / "_qwen3_asr"
        spec = importlib.util.spec_from_file_location(
            name, pkg / "__init__.py", submodule_search_locations=[str(pkg)]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return name


def _parse_quantize(spec: str | None) -> dict:
    """`quantize` = "enc:<s>,conv:<s>,dec:<s>,dec[a-b]:<s>,head:<s>,emb:<s>"; unset parts stay fp32.

    `enc` covers the tower's linears; `conv:8da8w` runs its conv stem as
    dynamically quantized int8 conv2d.

    `dec[a-b]` overrides `dec` for decoder layers a..b. The first decoder layer
    does not take 4 bits: the model then loses its `language <Name><asr_text>`
    header on some clips and derails (see NOTES.md). `head` is the 151936-wide
    output projection.
    """
    recipe = {"enc": "none", "conv": "none", "dec": "none", "head": "none", "emb": "none", "layers": {}}
    for item in (spec or "").split(","):
        if not item.strip():
            continue
        part, scheme = item.strip().split(":")
        if part.startswith("dec[") and part.endswith("]"):
            lo, hi = part[4:-1].split("-")
            recipe["layers"].update({i: scheme for i in range(int(lo), int(hi) + 1)})
        elif part in recipe:
            recipe[part] = scheme
        else:
            raise ValueError(f"unknown quantize part '{part}' in '{spec}'")
    return recipe


def _linear_config(scheme: str):
    """torchao config for "8da4w-g<N>" / "8da8w-pc" / "8da8w-g<N>"."""
    from torchao.quantization import MappingType
    from torchao.quantization.granularity import PerAxis, PerGroup
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig

    act_w, _, gran = scheme.partition("-")
    dtypes = {"8da4w": torch.int4, "8da8w": torch.int8}
    granularity = PerAxis(0) if gran == "pc" else PerGroup(int(gran[1:]))
    return Int8DynamicActivationIntxWeightConfig(
        weight_dtype=dtypes[act_w],
        weight_granularity=granularity,
        weight_mapping_type=MappingType.SYMMETRIC,
    )


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.xnnpack.partition.config.xnnpack_config import ConfigPrecisionType
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
    tf = importlib.import_module(f"{pkg}.transforms")
    passes = importlib.import_module(f"{pkg}.passes")

    variant = config["variant"]
    recipe = _parse_quantize(variant.get("quantize"))
    max_context = q.MAX_CONTEXT_LEN
    # Builder convention: the dynamic token dim is bounded one below the cache.
    max_prefill = max_context - 1

    m = q.Qwen3Asr()
    params = m.text_params()

    encoder = tf.xnnpack_audio_encoder(m.audio_encoder())
    if recipe["enc"] != "none":
        quantize_(encoder, _linear_config(recipe["enc"]))
    ep_pre = ex.export_preprocess(m.preprocessor())
    ep_enc = ex.export_audio_encoder(encoder, dynamic=True, quantize_convs=recipe["conv"] == "8da8w")

    text = m.text_model(
        use_kv_cache=True,
        use_sdpa_with_kv_cache_op=True,
        enable_dynamic_shape=True,
        max_seq_len=max_context,
        max_context_len=max_context,
    )
    text = tf.xnnpack_text_model(text, params["head_dim"])
    text = replace_kv_cache_with_custom_kv_cache(text)
    text = replace_sdpa_with_custom_op(text)
    def dec_scheme(fqn: str) -> str:
        if fqn == "output":
            return recipe["head"]
        if fqn.startswith("layers."):
            return recipe["layers"].get(int(fqn.split(".")[1]), recipe["dec"])
        return recipe["dec"]

    linears = [fqn for fqn, mod in text.named_modules() if isinstance(mod, torch.nn.Linear)]
    for scheme in sorted({dec_scheme(fqn) for fqn in linears} - {"none"}):
        quantize_(
            text,
            _linear_config(scheme),
            filter_fn=lambda mod, fqn, s=scheme: isinstance(mod, torch.nn.Linear) and dec_scheme(fqn) == s,
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
    if recipe["emb"] != "none":  # "<bits>w-g<group>", e.g. "8w-g64"
        bits, group = recipe["emb"].split("w-g")
        bits, group = int(bits), int(group)
        holder = torch.nn.Module()
        holder.embedding = emb
        emb = EmbeddingQuantHandler(holder, bitwidth=bits, group_size=group, packed=False).quantized_model().embedding
    tok = Dim("tokens", min=1, max=max_context)
    with torch.no_grad():
        ep_tok = torch.export.export(
            q.TokenEmbedding(emb), (torch.zeros(1, 8, dtype=torch.int64),), dynamic_shapes=({1: tok},), strict=True
        )

    schema = ex.model_schema(
        dim=params["dim"], vocab=params["vocab_size"], max_context=max_context,
        max_prefill=max_prefill, dynamic_encoder=True,
    )
    consts = dict(variant.get("constants", {}))
    consts["get_model_schema"] = schema

    dq_first = [
        # Dynamically quantized linears first, each in its own partition so
        # packed and unpacked weights are not held at once.
        XnnpackPartitioner(config_precisions=ConfigPrecisionType.DYNAMIC_QUANT, per_op_mode=True),
        XnnpackPartitioner(),
    ]
    program = to_edge_transform_and_lower(
        {"preprocess": ep_pre, "audio_encoder": ep_enc, "token_embedding": ep_tok, "text_decoder": ep_dec},
        partitioner={
            "preprocess": [XnnpackPartitioner()],
            "audio_encoder": dq_first,
            "token_embedding": [XnnpackPartitioner()],
            "text_decoder": dq_first,
        },
        transform_passes={
            "preprocess": [AddmmToLinearTransform()],
            "audio_encoder": [AddmmToLinearTransform()],
            "token_embedding": [AddmmToLinearTransform()],
            "text_decoder": [AddmmToLinearTransform(), passes.RemoveInversePermutePairs()],
        },
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
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
