import glob
import importlib
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


def _load_text_decoder():
    pkg_init = _HERE / "_text_decoder" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "gemma4_text_decoder",
        pkg_init,
        submodule_search_locations=[str(_HERE / "_text_decoder")],
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

    gemma4_config = Gemma4Config.from_config("e2b")
    gemma4_config.use_kv_cache = True
    gemma4_config.max_seq_len = CONTEXT_LEN
    gemma4_config.enable_dynamic_shape = True
    gemma4_config.use_custom_sdpa = False

    torch_dtype = torch.bfloat16
    logger.info("Building Gemma4 E2B text decoder (bf16)...")
    model_wrapper = Gemma4Model(
        config=gemma4_config, checkpoint_path=checkpoint_path, dtype=torch_dtype
    )
    model = model_wrapper.get_eager_model()
    model.eval()

    logger.info("Applying MLX source transformations (YOCO-preserving)...")
    mlx_source_transformations(model, dtype=torch_dtype, compile_friendly=True)

    logger.info(f"Quantizing: qlinear=4w qembedding=4w gs={GROUP_SIZE}")
    quantize_model_(
        model,
        qlinear_config="4w",
        qlinear_group_size=GROUP_SIZE,
        qembedding_config="4w",
        qembedding_group_size=GROUP_SIZE,
    )
    model.eval()

    max_prefill = gemma4_config.sliding_window
    example_seq_len = max_prefill - 1
    input_ids = torch.ones((1, example_seq_len), dtype=torch.long)
    input_pos = torch.arange(example_seq_len, dtype=torch.long)
    seq_len_dim = Dim("seq_len", min=1, max=max_prefill - 1)
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "input_pos": {0: seq_len_dim},
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
                strict=True,
            )

    llm_metadata = {
        "get_eos_ids": [1, 106, 50],
        "get_bos_id": 2,
        "get_max_seq_len": int(CONTEXT_LEN),
        "get_max_context_len": int(CONTEXT_LEN),
        "use_kv_cache": True,
        "enable_dynamic_shape": True,
    }
    logger.info(f"Embedding LLM runner metadata: {llm_metadata}")

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
