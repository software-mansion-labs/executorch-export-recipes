import glob
import importlib
import importlib.util
import json
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


def _lean_load():
    spec = importlib.util.spec_from_file_location(
        "gemma4_lean_load", _HERE.parent.parent / "gemma4" / "_lean_load.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.lean_load


def _quantize_embedding_in_chunks(embedding, group_size, rows_per_chunk=16384):
    from executorch.extension.llm.export.quantize import _make_embedding_config
    from torchao.quantization.quant_api import quantize_

    config = _make_embedding_config("4w", group_size)
    weight = embedding.weight.detach()
    parts = []
    for start in range(0, weight.shape[0], rows_per_chunk):
        chunk = torch.nn.Embedding.from_pretrained(
            weight[start : start + rows_per_chunk].clone(), freeze=True
        )
        quantize_(chunk, config, filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding))
        parts.append(chunk.weight)
    names, meta = parts[0].__tensor_flatten__()
    inner = {n: torch.cat([getattr(p, n) for p in parts]) for n in names}
    full = type(parts[0]).__tensor_unflatten__(inner, meta, weight.shape, None)
    embedding.weight = torch.nn.Parameter(full, requires_grad=False)


def _build_text_model_lean(checkpoint_path: str, max_seq_len: int, group_size: int = 64):
    from executorch.backends.mlx.llm.quantization import quantize_model_
    from gemma4_omni_pkg.text_decoder.gemma4_config import Gemma4Config
    from gemma4_omni_pkg.text_decoder.gemma4_model import Gemma4Model
    from gemma4_omni_pkg.text_decoder.mlx_source_transformations import mlx_source_transformations

    config = Gemma4Config.from_config("e2b")
    config.use_kv_cache = True
    config.max_seq_len = max_seq_len
    config.enable_dynamic_shape = True
    config.use_custom_sdpa = False

    logger.info("Lean loading eager model (bf16)...")
    lean_loader = _lean_load()
    with lean_loader():
        mw = Gemma4Model(config=config, checkpoint_path=checkpoint_path, dtype=torch.bfloat16)
    model = mw.get_eager_model()
    model.eval()

    logger.info("Applying mlx source transformations...")
    mlx_source_transformations(model, dtype=torch.bfloat16, compile_friendly=False)

    logger.info(f"Quantizing text decoder: qlinear=4w qembedding=4w gs={group_size}")
    parent_name, _, attr = next(
        name for name, _ in model.named_modules() if name.endswith("embed_tokens_per_layer")
    ).rpartition(".")
    parent = model.get_submodule(parent_name)
    per_layer = getattr(parent, attr)
    setattr(parent, attr, torch.nn.Identity())
    quantize_model_(
        model,
        qlinear_config="4w",
        qlinear_group_size=group_size,
        qembedding_config="4w",
        qembedding_group_size=group_size,
    )
    _quantize_embedding_in_chunks(per_layer, group_size)
    setattr(parent, attr, per_layer)
    model.eval()
    return model, config


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
    from exporter.schema import load_and_validate_schema

    _load_gemma4_pkg()

    v3_mod = importlib.import_module("gemma4_omni_pkg.export_gemma4_omni_mlx_v3")
    omni_mod = importlib.import_module("gemma4_omni_pkg.export_gemma4_omni_mlx")

    _build_vision_runner_ep = v3_mod._build_vision_runner_ep
    TokenEmbeddingModule = omni_mod.TokenEmbeddingModule
    TextDecoderModule = omni_mod.TextDecoderModule

    class F32TokenEmbeddingModule(TokenEmbeddingModule):
        def forward(self, input_ids: torch.Tensor):
            embeds, ple = super().forward(input_ids)
            return embeds.to(torch.float32), ple.to(torch.float32)

    class F32TextDecoderModule(TextDecoderModule):
        def forward(self, inputs_embeds: torch.Tensor, ple_tok: torch.Tensor, input_pos: torch.Tensor):
            if inputs_embeds.dtype != torch.bfloat16:
                inputs_embeds = inputs_embeds.to(torch.bfloat16)
            if ple_tok.dtype != torch.bfloat16:
                ple_tok = ple_tok.to(torch.bfloat16)
            logits = super().forward(inputs_embeds, ple_tok, input_pos)
            return logits.to(torch.float32)

    class PatchedRunnerAudioEncoder(torch.nn.Module):
        def __init__(self, speech, audio_with_proj):
            super().__init__()
            self.speech = speech
            self.audio = audio_with_proj

        def forward(self, waveform: torch.Tensor, num_valid_samples: torch.Tensor):
            num_valid = num_valid_samples.squeeze()
            mel = self.speech(waveform[0])
            mel = mel.reshape(-1, mel.shape[-1])
            mel = torch.nn.functional.pad(mel, (0, 0, 0, 1))
            n_frames = mel.shape[0]
            valid_frames = (num_valid // v3_mod.MEL_HOP - 1).clamp(min=1)
            frame_valid = torch.arange(n_frames) < valid_frames
            mel = mel * frame_valid.unsqueeze(-1).float()
            embeds, mask = self.audio(mel.unsqueeze(0), frame_valid.unsqueeze(0))
            return embeds.to(torch.float32), mask.to(torch.float32)

    def _build_patched_audio_runner_ep(checkpoint_path: str, group_size: int):
        from gemma4_omni_pkg.export_gemma4 import (
            _HFAudioEncoderWithProjection,
            _load_hf_audio_encoder,
            _patch_hf_audio_for_export,
        )
        from gemma4_omni_pkg.quant_utils import apply_linear_quantization
        from gemma4_omni_pkg.speech_transform import Gemma4SpeechTransformModel

        _patch_hf_audio_for_export()
        audio_with_proj = _load_hf_audio_encoder(checkpoint_path)
        assert isinstance(audio_with_proj, _HFAudioEncoderWithProjection)
        audio_with_proj = apply_linear_quantization(
            audio_with_proj, "8da4w", group_size=group_size
        )
        audio_with_proj.eval()

        kwargs = {}
        processor_path = os.path.join(checkpoint_path, "processor_config.json")
        if os.path.exists(processor_path):
            with open(processor_path) as f:
                audio_fe = json.load(f).get("feature_extractor", {})
            field_map = {
                "sampling_rate": "sample_rate",
                "feature_size": "n_mels",
                "min_frequency": "f_min",
                "max_frequency": "f_max",
                "mel_floor": "mel_floor",
                "input_scale_factor": "input_scale_factor",
                "frame_length": "frame_length",
                "hop_length": "hop_length",
            }
            for hf_key, our_key in field_map.items():
                if hf_key in audio_fe:
                    kwargs[our_key] = audio_fe[hf_key]
        speech = Gemma4SpeechTransformModel(**kwargs).eval()
        assert speech.hop_length == v3_mod.MEL_HOP

        audio_model = PatchedRunnerAudioEncoder(speech, audio_with_proj).eval()

        n_trace = v3_mod.BLOCK_SAMPLES * v3_mod.AUDIO_K_TRACE
        waveform = torch.zeros(1, n_trace, dtype=torch.float32)
        num_valid = torch.tensor([n_trace], dtype=torch.long)
        k_dim = torch.export.Dim("audio_k", min=v3_mod.AUDIO_K_MIN, max=v3_mod.AUDIO_K_MAX)
        with torch.no_grad():
            audio_ep = torch.export.export(
                audio_model,
                (waveform, num_valid),
                dynamic_shapes={
                    "waveform": {1: v3_mod.BLOCK_SAMPLES * k_dim},
                    "num_valid_samples": None,
                },
                strict=False,
            )
        return audio_model, audio_ep

    checkpoint_path = _resolve_checkpoint(None)
    out_path = str(Path(out_dir) / config["variant"]["file"])

    model, gemma_config = _build_text_model_lean(checkpoint_path, MAX_SEQ_LEN, GROUP_SIZE)

    tok_mod = F32TokenEmbeddingModule(model.model.self_decoder, gemma_config).eval()
    dec_mod = F32TextDecoderModule(model).eval()

    H = gemma_config.hidden_size
    L = gemma_config.num_hidden_layers
    P = gemma_config.hidden_size_per_layer_input
    window = gemma_config.sliding_window
    s_ex = min(MAX_SEQ_LEN, window) - 1

    ids = torch.ones((1, s_ex), dtype=torch.long)
    seq = Dim("seq", min=1, max=s_ex)
    with torch.no_grad():
        ep_tok = torch.export.export(
            tok_mod, (ids,), dynamic_shapes={"input_ids": {1: seq}}, strict=True
        )

    embeds = torch.randn(1, s_ex, H, dtype=torch.float32)
    ple = torch.randn(1, s_ex, L, P, dtype=torch.float32)
    pos = torch.arange(s_ex, dtype=torch.long)
    seq2 = Dim("seq2", min=1, max=s_ex)
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
    _, ep_aud = _build_patched_audio_runner_ep(checkpoint_path, GROUP_SIZE)

    # LLM runner constants come from the config (single source of truth).
    llm_metadata = dict(config["variant"].get("constants", {}))
    if not llm_metadata:
        for mname, m in config["variant"].get("methods", {}).items():
            if not (isinstance(m, dict) and "inputs" in m and "outputs" in m):
                llm_metadata[mname] = m

    schema_json = load_and_validate_schema(
        _HERE.parent / "schema.json",
        replacements={
            "seq_max": str(s_ex),
            "vocab_size": str(llm_metadata.get("get_vocab_size", 262144)),
        },
    )
    llm_metadata["get_model_schema"] = schema_json

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
