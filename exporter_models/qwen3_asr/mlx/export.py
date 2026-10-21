import importlib
import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# int4 decoder weights at group 32: at 64 the first generated token loses
# confidence on some clips (see NOTES.md). The embedding keeps group 64.
LINEAR_GROUP_SIZE = 32
EMBEDDING_GROUP_SIZE = 64
# Decoder layers kept at 8 bits. Layer 0 does not take 4 bits: the model then
# drops its `language <Name><asr_text>` header on some clips and derails.
INT8_LAYERS = (0, 1)


def _load_pkg() -> str:
    name = "qwen3_asr_mlx_pkg"
    if name not in sys.modules:
        pkg = _MODEL_DIR / "_qwen3_asr"
        spec = importlib.util.spec_from_file_location(
            name, pkg / "__init__.py", submodule_search_locations=[str(pkg)]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return name


class _CastIO(torch.nn.Module):
    """Runs `inner(x, aux)` in `dtype` behind an fp32 interface.

    MLX has no compile-time precision switch: the compute dtype is the torch
    model's dtype. The RNE task exchanges fp32 tensors between methods, so the
    float input is cast down on the way in and the output back up. `aux` is the
    integer second input (`start_pos`).
    """

    def __init__(self, inner: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.inner = inner.to(dtype)
        self.dtype = dtype

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.inner(x.to(self.dtype), aux).float()


class _CastOut(torch.nn.Module):
    """`_CastIO` for the token embedding, whose only input is integer ids."""

    def __init__(self, inner: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.inner = inner.to(dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.inner(tokens).float()


def export(config: dict, out_dir: str) -> str:
    import executorch.backends.mlx.llm.et_attention  # noqa: F401  registers attention_type="mlx"
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.backends.mlx.llm.quantization import quantize_model_
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
    from torch.export import Dim
    from torch.nn.attention import SDPBackend, sdpa_kernel

    pkg = _load_pkg()
    q = importlib.import_module(pkg)
    ex = importlib.import_module(f"{pkg}.exportable")

    variant = config["variant"]
    max_context = q.MAX_CONTEXT_LEN
    max_prefill = max_context - 1

    m = q.Qwen3Asr()
    params = m.text_params()

    # The STFT front end stays on XNNPACK, which every iOS build links. The
    # tower goes to Core ML (fp16) on one static 8 s window, n_valid masking
    # the tail: on an iPhone 16 it runs 2.4x faster there than on MLX
    # (NOTES.md), so this file needs both the MLX and the Core ML backend.
    ep_pre = ex.export_preprocess(m.preprocessor())
    ep_enc = ex.export_audio_encoder(m.audio_encoder().eval(), dynamic=False)

    # attention_type="mlx" builds MLX's own attention + KV cache inside
    # construct_transformer, so none of the llama custom-op source transforms
    # apply (same as lfm2_5_vl/mlx).
    text = m.text_model(
        use_kv_cache=True,
        use_sdpa_with_kv_cache_op=False,
        enable_dynamic_shape=True,
        attention_type="mlx",
        max_seq_len=max_context,
        max_context_len=max_context,
    ).to(torch.bfloat16)
    emb = m.token_embedding().to(torch.bfloat16)
    if variant["quantized"]:
        for i, layer in enumerate(text.layers):
            if i in INT8_LAYERS:
                quantize_model_(layer, qlinear_config="8w")
            else:
                quantize_model_(layer, qlinear_config="4w", qlinear_group_size=LINEAR_GROUP_SIZE)
        head = torch.nn.Sequential(text.output)
        quantize_model_(head, qlinear_config="4w", qlinear_group_size=LINEAR_GROUP_SIZE)
        text.output = head[0]
        holder = torch.nn.Sequential(emb)
        quantize_model_(holder, qembedding_config="4w", qembedding_group_size=EMBEDDING_GROUP_SIZE)
        emb = holder[0]

    decoder = _CastIO(q.TextDecoder(text), torch.bfloat16).eval()
    seq = Dim("seq", min=1, max=max_prefill)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        ep_dec = torch.export.export(
            decoder,
            (torch.randn(1, 8, params["dim"]), torch.tensor([0], dtype=torch.int64)),
            dynamic_shapes=({1: seq}, None),
            strict=False,
        )

    tok = Dim("tokens", min=1, max=max_context)
    with torch.no_grad():
        ep_tok = torch.export.export(
            _CastOut(q.TokenEmbedding(emb), torch.bfloat16).eval(),
            (torch.zeros(1, 8, dtype=torch.int64),),
            dynamic_shapes=({1: tok},),
            strict=False,
        )

    schema = ex.model_schema(
        dim=params["dim"], vocab=params["vocab_size"], max_context=max_context,
        max_prefill=max_prefill, dynamic_encoder=False,
    )
    consts = dict(variant.get("constants", {}))
    consts["get_model_schema"] = schema

    mlx_passes = get_default_passes()
    tower_specs = CoreMLBackend.generate_compile_specs(
        compute_unit=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    program = to_edge_transform_and_lower(
        {"preprocess": ep_pre, "audio_encoder": ep_enc, "token_embedding": ep_tok, "text_decoder": ep_dec},
        partitioner={
            "preprocess": [XnnpackPartitioner()],
            "audio_encoder": [CoreMLPartitioner(compile_specs=tower_specs)],
            "token_embedding": [MLXPartitioner()],
            "text_decoder": [MLXPartitioner()],
        },
        transform_passes={
            "preprocess": [],
            "audio_encoder": [],
            "token_embedding": mlx_passes,
            "text_decoder": mlx_passes,
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
