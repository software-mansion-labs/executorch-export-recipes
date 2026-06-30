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
MAX_SEQ_LEN = 2048

logger = logging.getLogger(__name__)


def _load_gemma4_pkg():
    pkg_init = _HERE.parent / "_gemma4" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "gemma4_omni_pkg",
        pkg_init,
        submodule_search_locations=[str(_HERE.parent / "_gemma4")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemma4_omni_pkg"] = mod
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
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner,
        XnnpackPartitioner,
    )
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim

    _load_gemma4_pkg()

    v3_mod = importlib.import_module("gemma4_omni_pkg.export_gemma4_omni_mlx_v3")
    omni_mod = importlib.import_module("gemma4_omni_pkg.export_gemma4_omni_mlx")

    _build_vision_runner_ep = v3_mod._build_vision_runner_ep
    _build_audio_runner_ep = v3_mod._build_audio_runner_ep
    TokenEmbeddingModule = omni_mod.TokenEmbeddingModule
    TextDecoderModule = omni_mod.TextDecoderModule
    _build_text_model = omni_mod._build_text_model

    checkpoint_path = _resolve_checkpoint(None)
    out_path = str(Path(out_dir) / config["variant"]["file"])

    model, gemma_config = _build_text_model(checkpoint_path, MAX_SEQ_LEN, GROUP_SIZE)

    tok_mod = TokenEmbeddingModule(model.model.self_decoder, gemma_config).eval()
    dec_mod = TextDecoderModule(model).eval()

    H = gemma_config.hidden_size
    L = gemma_config.num_hidden_layers
    P = gemma_config.hidden_size_per_layer_input
    window = gemma_config.sliding_window
    s_ex = min(MAX_SEQ_LEN, window) - 1

    ids = torch.ones((1, s_ex), dtype=torch.long)
    seq = Dim("seq", min=1, max=MAX_SEQ_LEN)
    with torch.no_grad():
        ep_tok = torch.export.export(
            tok_mod, (ids,), dynamic_shapes={"input_ids": {1: seq}}, strict=True
        )

    embeds = torch.randn(1, s_ex, H)
    ple = torch.randn(1, s_ex, L, P)
    pos = torch.arange(s_ex, dtype=torch.long)
    seq2 = Dim("seq2", min=1, max=window - 1)
    with torch.no_grad(), torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.MATH]
    ):
        ep_dec = torch.export.export(
            dec_mod,
            (embeds, ple, pos),
            dynamic_shapes={
                "inputs_embeds": {1: seq2},
                "ple_tok": {1: seq2},
                "input_pos": {0: seq2},
            },
            strict=True,
        )

    logger.info("Building vision encoder ep (runner contract, MLX int4)...")
    _, ep_vis = _build_vision_runner_ep(checkpoint_path, GROUP_SIZE)
    logger.info("Building audio encoder ep (runner contract, XNNPACK 8da4w)...")
    _, ep_aud = _build_audio_runner_ep(checkpoint_path, GROUP_SIZE)

    llm_metadata = {
        "get_eos_ids": [1, 106, 50],
        "get_bos_id": 2,
        "get_vocab_size": int(gemma_config.vocab_size),
        "get_max_seq_len": int(min(MAX_SEQ_LEN, window)) - 1,
        "get_max_context_len": int(MAX_SEQ_LEN),
        "use_kv_cache": True,
        "enable_dynamic_shape": True,
        "has_ple": True,
        "image_placeholder_id": int(gemma_config.image_token_id),
    }

    programs = {
        "token_embedding": ep_tok,
        "text_decoder": ep_dec,
        "vision_encoder": ep_vis,
        "audio_encoder": ep_aud,
    }

    mlx = [MLXPartitioner()]
    xnn = [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]
    partitioners = {
        "token_embedding": mlx,
        "text_decoder": mlx,
        "vision_encoder": mlx,
        "audio_encoder": xnn,
    }
    mlx_passes = get_default_passes()
    transform_passes = {
        "token_embedding": mlx_passes,
        "text_decoder": mlx_passes,
        "vision_encoder": mlx_passes,
        "audio_encoder": [],
    }

    logger.info("Lowering: MLX (text/vision) + XNNPACK (audio)...")
    edge = exir.to_edge_transform_and_lower(
        programs,
        transform_passes=transform_passes,
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        constant_methods=llm_metadata,
    )
    et = edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
        )
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(et.buffer)
    logger.info(f"Saved {out_path} ({len(et.buffer) / 1024 / 1024:.1f} MB)")
    return out_path
