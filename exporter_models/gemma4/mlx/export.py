import glob
import importlib
import json
import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent

GROUP_SIZE = 64
CONTEXT_LEN = 2048

logger = logging.getLogger(__name__)


def _lean_load():
    spec = importlib.util.spec_from_file_location("gemma4_lean_load", _HERE.parent / "_lean_load.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.lean_load


def _quantize_embedding_in_chunks(embedding, group_size, rows_per_chunk=16384):
    """4w-quantizes one embedding table a block of rows at a time.

    Bit-identical to `quantize_model_`'s "4w" embedding path (same torchao
    config; groups run along each row, so rows quantize independently), but its
    peak is one chunk instead of the whole table. Gemma 4's per-layer embedding
    table is 262144 x 8960 (2.35B params), and quantizing it in one go does not
    fit in 16 GB.
    """
    from executorch.extension.llm.export.quantize import _make_embedding_config
    from torchao.quantization.quant_api import quantize_

    config = _make_embedding_config("4w", group_size)
    weight = embedding.weight.detach()
    parts = []
    for start in range(0, weight.shape[0], rows_per_chunk):
        chunk = torch.nn.Embedding.from_pretrained(
            weight[start:start + rows_per_chunk].clone(), freeze=True
        )
        quantize_(chunk, config, filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding))
        parts.append(chunk.weight)
    names, meta = parts[0].__tensor_flatten__()
    inner = {n: torch.cat([getattr(p, n) for p in parts]) for n in names}
    full = type(parts[0]).__tensor_unflatten__(inner, meta, weight.shape, None)
    embedding.weight = torch.nn.Parameter(full, requires_grad=False)


def _load_text_decoder():
    pkg_init = _HERE.parent / "_text_decoder" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "gemma4_text_decoder",
        pkg_init,
        submodule_search_locations=[str(_HERE.parent / "_text_decoder")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemma4_text_decoder"] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_checkpoint(arg):
    if arg and os.path.exists(arg):
        return os.path.abspath(arg)
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/"
            "snapshots/*/model.safetensors"
        )
    )
    if not hits:
        raise SystemExit(
            "Gemma needs a checkpoint dir with model.safetensors. "
            "None found in the HF cache. Pass checkpoint_path explicitly."
        )
    return os.path.dirname(hits[0])


def export(config: dict, out_dir: str) -> str:
    from executorch import exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.llm.quantization import quantize_model_
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim

    td = _load_text_decoder()
    Gemma4Config = td.Gemma4Config
    Gemma4Model = td.Gemma4Model
    mlx_source_transformations_mod = importlib.import_module(
        "gemma4_text_decoder.mlx_source_transformations"
    )
    mlx_source_transformations = mlx_source_transformations_mod.mlx_source_transformations

    checkpoint_path = _resolve_checkpoint(None)
    out_path = str(Path(out_dir) / config["variant"]["file"])
    consts = dict(config["variant"].get("constants", {}))
    max_ctx = consts.get("get_max_context_len", CONTEXT_LEN)

    gemma4_config = Gemma4Config.from_config("e2b")
    gemma4_config.use_kv_cache = True
    gemma4_config.max_seq_len = max_ctx
    gemma4_config.enable_dynamic_shape = True
    gemma4_config.use_custom_sdpa = False

    torch_dtype = torch.bfloat16
    logger.info("Building Gemma4 E2B text decoder (bf16)...")
    # Built and loaded straight at bf16: the model's own fp32 construction plus a
    # full state dict does not fit in 16 GB.
    with _lean_load()():
        model_wrapper = Gemma4Model(
            config=gemma4_config, checkpoint_path=checkpoint_path, dtype=torch_dtype
        )
    model = model_wrapper.get_eager_model()
    model.eval()

    logger.info("Applying MLX source transformations (YOCO-preserving)...")
    # compile_friendly=False: the compile-friendly path calls mlx::rope_t, which
    # the ExecuTorch 1.4.1 fork (and so the RNE runtime) does not register; the
    # standard path uses mlx::rope / kv_cache_update / custom_sdpa, which it does.
    mlx_source_transformations(model, dtype=torch_dtype, compile_friendly=False)

    logger.info(f"Quantizing: qlinear=4w qembedding=4w gs={GROUP_SIZE}")
    # The per-layer embedding table is quantized on its own, in row chunks; it
    # is swapped out while quantize_model_ handles everything else, so it is
    # not quantized twice.
    parent_name, _, attr = next(
        name for name, _ in model.named_modules() if name.endswith("embed_tokens_per_layer")
    ).rpartition(".")
    parent = model.get_submodule(parent_name)
    per_layer = getattr(parent, attr)
    setattr(parent, attr, torch.nn.Identity())
    quantize_model_(
        model,
        qlinear_config="4w",
        qlinear_group_size=GROUP_SIZE,
        qembedding_config="4w",
        qembedding_group_size=GROUP_SIZE,
    )
    _quantize_embedding_in_chunks(per_layer, GROUP_SIZE)
    setattr(parent, attr, per_layer)
    model.eval()

    # A prefill chunk may not exceed the sliding window, so the token axis stays
    # bounded at sliding_window - 1 whatever the context. input_pos is the
    # runner's [1] start position; the model expands it to per-token positions.
    max_prefill = gemma4_config.sliding_window
    example_seq_len = max_prefill - 1
    input_ids = torch.ones((1, example_seq_len), dtype=torch.long)
    input_pos = torch.zeros((1,), dtype=torch.long)
    seq_len_dim = Dim("seq_len", min=1, max=max_prefill - 1)
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "input_pos": {},
    }
    logger.info(f"Export 2-input forward; seq_len dynamic in [1, {max_prefill - 1}]")

    logger.info("torch.export ...")
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        with torch.no_grad():
            ep = torch.export.export(
                model,
                (input_ids,),
                kwargs={"input_pos": input_pos},
                dynamic_shapes=dynamic_shapes,
                # Non-strict: the MLX attention transform caches its mask by
                # id(input_pos), which strict (dynamo) tracing cannot compare.
                strict=False,
            )

    # LLM runner constants come from the config (single source of truth); the
    # chunk size and the schema follow the graph's actual token bound.
    vocab_size = consts.get("get_vocab_size", gemma4_config.vocab_size)
    model_schema = {
        "forward": {
            "inputs": [
                {
                    "kind": "Tensor",
                    "dtype": "int64",
                    "shape": [
                        {"kind": "constant", "value": 1},
                        {"kind": "range", "range": {"min": 1, "max": max_prefill - 1, "step": 1}},
                    ],
                },
                {"kind": "Tensor", "dtype": "int64", "shape": [{"kind": "constant", "value": 1}]},
            ],
            "outputs": [
                {
                    "kind": "Tensor",
                    "dtype": "float32",
                    "shape": [
                        {"kind": "constant", "value": 1},
                        {"kind": "constant", "value": vocab_size},
                    ],
                }
            ],
            "runtimeConstraints": [],
        }
    }
    root_schema_path = _HERE.parent.parent / "model.schema.json"
    if root_schema_path.exists():
        import jsonschema

        jsonschema.validate(instance=model_schema, schema=json.loads(root_schema_path.read_text()))

    llm_metadata = {
        k: v for k, v in consts.items() if k not in ("get_n_layers", "use_sdpa_with_kv_cache")
    }
    llm_metadata.update(
        get_max_seq_len=max_prefill,
        get_max_context_len=max_ctx,
        get_vocab_size=vocab_size,
        use_kv_cache=True,
        enable_dynamic_shape=True,
        get_model_schema=json.dumps(model_schema),
    )
    logger.info(
        "Embedding LLM runner metadata: %s",
        {k: v for k, v in llm_metadata.items() if k != "get_model_schema"},
    )

    logger.info("Lowering to MLX backend...")
    edge_program = exir.to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=llm_metadata,
    )

    logger.info("to_executorch ...")
    et_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
        )
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(et_program.buffer)
    logger.info(f"Saved {out_path} ({len(et_program.buffer) / 1024 / 1024:.1f} MB)")
    return out_path
