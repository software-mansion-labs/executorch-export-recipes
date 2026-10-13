"""Gemma 4 E2B multimodal (omni) model, XNNPACK backend, int4.

4-method PTE matching LLM_GEMMA_SPEC in react-native-executorch:
  - token_embedding: input_ids [1, S] (int64) -> inputs_embeds [1, S, 1536] (fp32), ple_tokens [1, S, 35, 256] (fp32)
  - text_decoder: embeds [1, S, 1536] (fp32), ple_tokens [1, S, 35, 256] (fp32), input_pos [S] (int64) -> logits [1, 1, 262144] (fp32)
  - vision_encoder: pixel_values [1, 3, 448, 640] (fp32) -> embeds [1, 280, 1536] (fp32)
  - audio_encoder: waveform [1, 7680*k] (fp32), num_valid_samples [1] (int64) -> audio_embeds [1, 12*k, 1536] (fp32), audio_mask [1, 12*k] (fp32)
"""
import importlib
import importlib.util
import logging
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_OMNI_DIR = _HERE.parent
_GEMMA4_DIR = _OMNI_DIR.parent / "gemma4"
_EDGE_PKG = "gemma4_edge_omni"

logger = logging.getLogger(__name__)

# --- quantisation -----------------------------------------------------------
LINEAR_BITS = 4
LINEAR_GROUP_SIZE = 128
QUANTIZE_MAIN_EMBEDDINGS = True
EMBEDDING_GROUP_SIZE = 32
QUANTIZE_PLE_EMBEDDINGS = True
PLE_EMBEDDING_BITS = 4
PLE_EMBEDDING_GROUP_SIZE = 32
BASE_DTYPE = torch.float16
FP16_RETARGET = False
LOGITS_DTYPE = torch.float32

# --- sequence & shape bounds ------------------------------------------------
NUM_LAYERS = 35
NUM_KV_SHARED_LAYERS = 20
MAX_SEQ_LEN = 4048
PREFILL_LEN = 1024
TEXT_DECODER_PREFILL_LEN = 1024

VIS_H = 448
VIS_W = 640
POOL_K = 2

AUDIO_BLOCK_SAMPLES = 7680
AUDIO_K_MIN = 1
AUDIO_K_MAX = 62
AUDIO_SAMPLE_RATE = 16000

# --- attention / cache flags ------------------------------------------------
USE_BSHD_KV = True
USE_DYNAMIC_KV_SLICE = True
USE_BOOL_MASK = True
USE_JIT_MASK = True
USE_SHARED_RING_STATE = True
SKIP_REDUNDANT_ROPE_SLICE = True
USE_BROADCAST_GQA = True
USE_CUSTOM_SDPA = True
USE_RSQRT = True
INT8_KV_CACHE = True
RING_KV_CACHE = True
RETURN_LAST_ONLY = True


def _load_edge():
    """Path-load the vendored `_gemma4_edge` package under `_EDGE_PKG`."""
    if _EDGE_PKG in sys.modules:
        return sys.modules[_EDGE_PKG]
    root = _GEMMA4_DIR / "_gemma4_edge"
    spec = importlib.util.spec_from_file_location(
        _EDGE_PKG, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EDGE_PKG] = module
    spec.loader.exec_module(module)
    return module


def _quantize_from_bf16(pipeline) -> None:
    """Quantizes from the bf16 checkpoint, then casts what is left to fp16."""
    load = pipeline.load_truncated_from_safetensors
    quantize = pipeline._quantize

    def load_bf16(*args, **kwargs):
        kwargs["dtype"] = torch.bfloat16
        return load(*args, **kwargs)

    quant = importlib.import_module(f"{_EDGE_PKG}.export.quant")

    def quantize_then_fp16(model, recipe, **kwargs):
        model = quantize(model, recipe, **kwargs)
        quant.retarget_quantized_to_fp16_with_fp32_scales(model)
        for mod in model.modules():
            if quant._is_torchao_parametrized(mod):
                scale = mod.parametrizations.weight.original1
                scale.data = scale.data.to(BASE_DTYPE)
        return model

    pipeline.load_truncated_from_safetensors = load_bf16
    pipeline._quantize = quantize_then_fp16


def _adapt_multimodal(pipeline, wrappers):
    """Adapts wrappers and build_programs to match LLM_GEMMA_SPEC."""
    import functools
    from transformers.models.gemma4 import modeling_gemma4
    if hasattr(modeling_gemma4, "Gemma4AudioCausalConv1d"):
        conv_cls = modeling_gemma4.Gemma4AudioCausalConv1d
        cached = conv_cls.__dict__.get("left_pad")
        if isinstance(cached, functools.cached_property):
            conv_cls.left_pad = property(cached.func)

    orig_TokenEmbeddingExport = wrappers.TokenEmbeddingExport
    orig_TextDecoderExport = wrappers.TextDecoderExport
    orig_AudioEncoderWavExport = wrappers.AudioEncoderWavExport

    class F32TokenEmbeddingExport(orig_TokenEmbeddingExport):
        def forward(self, input_ids: torch.Tensor):
            embeds, ple = super().forward(input_ids)
            return embeds.to(torch.float32), ple.to(torch.float32)

    class F32TextDecoderExport(orig_TextDecoderExport):
        def forward(self, embeds: torch.Tensor, ple_tokens: torch.Tensor, input_pos: torch.Tensor):
            if embeds.dtype != BASE_DTYPE:
                embeds = embeds.to(BASE_DTYPE)
            if ple_tokens.dtype != BASE_DTYPE:
                ple_tokens = ple_tokens.to(BASE_DTYPE)
            return super().forward(embeds, ple_tokens, input_pos)

    class OmniAudioEncoderWavExport(orig_AudioEncoderWavExport):
        def forward(self, waveform: torch.Tensor, num_valid_samples: torch.Tensor):
            num_valid = num_valid_samples.squeeze()
            log_mel, mask = self.mel_frontend(waveform, num_valid)
            log_mel = F.pad(log_mel, (0, 0, 0, 1))
            mask = F.pad(mask, (0, 1), value=False)
            out = self.inner(log_mel, mask)
            mask_downsampled = mask[:, ::4].float()
            return out.to(torch.float32), mask_downsampled

    wrappers.TokenEmbeddingExport = F32TokenEmbeddingExport
    pipeline.TokenEmbeddingExport = F32TokenEmbeddingExport

    wrappers.TextDecoderExport = F32TextDecoderExport
    pipeline.TextDecoderExport = F32TextDecoderExport

    wrappers.AudioEncoderWavExport = OmniAudioEncoderWavExport
    pipeline.AudioEncoderWavExport = OmniAudioEncoderWavExport

    def _patched_build_programs(model, towers, recipe, verbose=True):
        from torch.export import Dim
        L = recipe.lengths
        vision, audio = towers
        programs = {}

        seq_dim = Dim("seq_len", min=1, max=L.prefill_len)
        text_dec_seq_dim = Dim(
            "text_dec_seq_len", min=1, max=L.effective_text_decoder_prefill_len
        )

        example_len = 8
        example_ids = torch.zeros(1, example_len, dtype=torch.long)
        example_pos = torch.arange(example_len, dtype=torch.long)

        token_embedder = F32TokenEmbeddingExport(model).eval()
        text_decoder = F32TextDecoderExport(model).eval()

        model.reset_caches()
        with torch.no_grad():
            example_embeds, example_ple = token_embedder(example_ids)

        programs["token_embedding"] = pipeline.torch_export(
            token_embedder,
            (example_ids,),
            strict=True,
            dynamic_shapes={"input_ids": {1: seq_dim}},
        )

        model.reset_caches()
        programs["text_decoder"] = pipeline.torch_export(
            text_decoder,
            (example_embeds, example_ple, example_pos),
            strict=True,
            dynamic_shapes={
                "embeds": {1: text_dec_seq_dim},
                "ple_tokens": {1: text_dec_seq_dim},
                "input_pos": {0: text_dec_seq_dim},
            },
        )
        model.reset_caches()

        if vision is not None:
            vision_tower, embed_vision = vision
            hf_style = pipeline._uses_hf_vision_tower(recipe)
            vision_mod = wrappers.VisionEncoderExport(
                vision_tower,
                embed_vision,
                vis_nx=L.patch_grid_nx,
                vis_ny=L.patch_grid_ny,
                patch_size=L.patch_size,
                pool_k=L.pool_k,
                max_soft_tokens=L.max_soft_tokens,
                encoder_style="layer_loop" if hf_style else "module",
                emit_ple_tokens=not hf_style,
                lm=None if hf_style else model,
                pad_token_id=0,
            ).eval()

            image_example = torch.zeros(1, 3, L.vision_h, L.vision_w, dtype=torch.float32)
            programs["vision_encoder"] = pipeline.torch_export(
                vision_mod, (image_example,), strict=True
            )

        if audio is not None:
            k_dim = Dim("audio_k", min=L.audio_k_min, max=L.audio_k_max)
            samples_dim = L.audio_block_samples * k_dim
            n_trace = L.audio_block_samples * 2
            waveform_example = torch.zeros(1, n_trace, dtype=torch.float32)
            num_valid_example = torch.tensor([n_trace], dtype=torch.long)
            programs["audio_encoder"] = pipeline.torch_export(
                audio,
                (waveform_example, num_valid_example),
                strict=True,
                dynamic_shapes={
                    "waveform": {1: samples_dim},
                    "num_valid_samples": None,
                },
            )

        return {m: programs[m] for m in recipe.method_names if m in programs}

    pipeline._build_programs = _patched_build_programs


def export(config: dict, out_dir: str) -> str:
    from executorch.runtime import Runtime  # noqa: F401
    import executorch.kernels.quantized  # noqa: F401
    import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401

    from exporter.schema import load_and_validate_schema

    _load_edge()
    edge_config = importlib.import_module(f"{_EDGE_PKG}.export.config")
    harness = importlib.import_module(f"{_EDGE_PKG}.export.harness")
    pipeline = importlib.import_module(f"{_EDGE_PKG}.export.pipeline")
    wrappers = importlib.import_module(f"{_EDGE_PKG}.export.wrappers")

    _adapt_multimodal(pipeline, wrappers)
    _quantize_from_bf16(pipeline)

    variant = config["variant"]
    recipe = harness.HarnessRecipe(
        name="xnnpack-omni",
        backend="xnnpack",
        modality="multimodal",
        validated_on_device=True,
        quant=edge_config.QuantSpec(
            linear_bits=LINEAR_BITS,
            linear_group_size=LINEAR_GROUP_SIZE,
            quantize_main_embeddings=QUANTIZE_MAIN_EMBEDDINGS,
            embedding_group_size=EMBEDDING_GROUP_SIZE,
            quantize_ple_embeddings=QUANTIZE_PLE_EMBEDDINGS,
            ple_embedding_bits=PLE_EMBEDDING_BITS,
            ple_embedding_group_size=PLE_EMBEDDING_GROUP_SIZE,
            base_dtype=BASE_DTYPE,
            fp16_retarget=FP16_RETARGET,
            logits_dtype=LOGITS_DTYPE,
            vision_quant="8da4w",
            vision_group_size=128,
            audio_quant="8da4w",
            audio_group_size=128,
        ),
        lengths=edge_config.LengthSpec(
            num_layers=NUM_LAYERS,
            num_kv_shared_layers=NUM_KV_SHARED_LAYERS,
            max_seq_len=variant.get("constants", {}).get("get_max_context_len", MAX_SEQ_LEN),
            prefill_len=PREFILL_LEN,
            text_decoder_prefill_len=TEXT_DECODER_PREFILL_LEN,
            vision_h=VIS_H,
            vision_w=VIS_W,
            pool_k=POOL_K,
            audio_block_samples=AUDIO_BLOCK_SAMPLES,
            audio_k_min=AUDIO_K_MIN,
            audio_k_max=AUDIO_K_MAX,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
        ),
        attn=edge_config.AttnFlags(
            use_bshd_kv=USE_BSHD_KV,
            use_dynamic_kv_slice=USE_DYNAMIC_KV_SLICE,
            use_bool_mask=USE_BOOL_MASK,
            use_jit_mask=USE_JIT_MASK,
            use_shared_ring_state=USE_SHARED_RING_STATE,
            skip_redundant_rope_slice=SKIP_REDUNDANT_ROPE_SLICE,
            use_broadcast_gqa=USE_BROADCAST_GQA,
            use_custom_sdpa=USE_CUSTOM_SDPA,
            use_rsqrt=USE_RSQRT,
            int8_kv_cache=INT8_KV_CACHE,
            ring_kv_cache=RING_KV_CACHE,
            return_last_only=RETURN_LAST_ONLY,
        ),
        constants=dict(variant.get("constants", {})),
    )
    harness.check_constants(recipe)

    schema_json = load_and_validate_schema(
        _OMNI_DIR / "schema.json",
        replacements={
            "seq_max": str(PREFILL_LEN),
            "vocab_size": str(variant.get("constants", {}).get("get_vocab_size", 262144)),
        },
    )

    checkpoint = harness.resolve_checkpoint()
    out_path = Path(out_dir) / variant["file"]
    logger.info("Gemma4 E2B multimodal / xnnpack / int4 from %s to %s", checkpoint, out_path)
    return str(
        pipeline.export(
            recipe,
            checkpoint=Path(checkpoint),
            out_path=out_path,
            schema_json=schema_json,
        )
    )
