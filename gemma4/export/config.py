"""Typed configuration for a Gemma4 export.

A `Recipe` is the complete description of one export: which backend, which
modalities, what quantisation, what sequence bounds, which attention flags.
`presets.py` holds four of them, matching the four PTEs that have actually been
built and shipped. The CLI resolves a preset and applies overrides on top.

Everything here is a frozen dataclass, so a Recipe can be printed, diffed and
compared. `Recipe.overridden(...)` returns a new Recipe rather than mutating.

Why the flags live in three places in the model, and therefore here:
the Gemma4 model reads configuration from (a) dataclass fields on
`Gemma4EdgeConfig`, (b) CLASS-level attributes on `Gemma4Attention` and
`RMSNorm`, and (c) INSTANCE attributes set per attention module. The class-level
ones must be set before the model is constructed; the instance-level ones after.
`AttnFlags` collects all three and `pipeline.py` applies each at the right
moment, so a caller never has to remember which is which.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

Backend = Literal["xnnpack", "vulkan"]
Modality = Literal["text", "multimodal"]


@dataclass(frozen=True)
class QuantSpec:
    """Quantisation scheme, per subsystem.

    The text decoder always uses 8da4w (int8 dynamic activations, int4 weights)
    for linears; weight-only int4 is NOT a valid alternative because
    XnnpackPartitioner const-folds dequantised weights back into its delegate
    blob and the size saving vanishes entirely (measured 1.00x, i.e. none).
    """

    # --- text decoder linears ---
    linear_bits: int = 4
    linear_group_size: int = 128
    linear_dynamic_activation: bool = True
    # q/k/v/o_proj are excluded in the multimodal recipes: bisected 2026-04-23,
    # quantising QKVO on top of the MLPs compounded drift far enough to flip the
    # sampled token (seed 10979 -> 9366). Left fp16 instead.
    skip_linear_name_suffixes: tuple[str, ...] = ()

    # --- embeddings ---
    quantize_main_embeddings: bool = True
    embedding_group_size: int = 32
    quantize_ple_embeddings: bool = True
    ple_embedding_bits: int = 4
    ple_embedding_group_size: int = 32

    # --- dtypes ---
    # Weight dtype the checkpoint is loaded at, before quantisation.
    base_dtype: torch.dtype = torch.float16
    # Vulkan only: rewrite torchao's quantised stack to fp16 while keeping the
    # scales in fp32. The Vulkan shaders are fp16; the scales must stay fp32 or
    # the dequant loses too much range. See quant.retarget_*.
    fp16_retarget: bool = False
    logits_dtype: torch.dtype = torch.float16

    # --- towers ---
    vision_quant: Literal["none", "8da4w"] = "none"
    vision_group_size: int = 128
    audio_quant: Literal["none", "8da4w"] = "none"
    audio_group_size: int = 128


@dataclass(frozen=True)
class LengthSpec:
    """Every sequence/shape bound baked into the graph.

    The critical, non-obvious relationship: the activation ARENA is sized by
    the prefill bounds, not by `max_seq_len`. `max_seq_len` only sizes the KV
    cache. Holding the prefill bound low while raising `max_seq_len` buys long
    generation cheaply; raising the prefill bound is what costs arena.
    """

    num_layers: int = 35
    num_kv_shared_layers: int = 20

    # KV cache depth == total generation budget.
    max_seq_len: int = 2048

    # Upper bound of the dynamic seq dim for the token-embedding / single
    # `forward` method. Also the example-input length used at trace time.
    prefill_len: int = 128

    # Upper bound of the dynamic seq dim for `text_decoder`, independent of
    # `prefill_len` in the multimodal recipes. None => same as prefill_len.
    # This is the knob that governs the text_decoder arena, and the one that
    # broke iter214: at 128 it is smaller than the 744 audio soft tokens the
    # audio encoder emits, and the runner does not bounds-check.
    text_decoder_prefill_len: Optional[int] = None

    # --- vision ---
    vision_h: int = 448
    vision_w: int = 640
    patch_size: int = 16
    pool_k: int = 2
    max_soft_tokens: int = 280

    # --- audio ---
    # Waveform length must be exactly audio_block_samples * k for k in range.
    audio_block_samples: int = 7680
    audio_k_min: int = 1
    audio_k_max: int = 62
    audio_soft_tokens_max: int = 744
    audio_sample_rate: int = 16000

    @property
    def effective_text_decoder_prefill_len(self) -> int:
        return (
            self.text_decoder_prefill_len
            if self.text_decoder_prefill_len is not None
            else self.prefill_len
        )

    @property
    def patch_grid_nx(self) -> int:
        return self.vision_w // self.patch_size

    @property
    def patch_grid_ny(self) -> int:
        return self.vision_h // self.patch_size

    @property
    def max_patches(self) -> int:
        return self.patch_grid_nx * self.patch_grid_ny

    @property
    def audio_sample_count_min(self) -> int:
        return self.audio_block_samples * self.audio_k_min

    @property
    def audio_sample_count_max(self) -> int:
        return self.audio_block_samples * self.audio_k_max


@dataclass(frozen=True)
class AttnFlags:
    """Attention / norm / cache flags.

    Grouped by where the model reads them from:
      class-level on Gemma4Attention: use_bshd_kv .. use_prebroadcast_rope
      instance-level per attention:   use_broadcast_gqa, use_custom_sdpa
      class-level on RMSNorm:         use_rsqrt  (NOTE: text and audio RMSNorm
                                      are SEPARATE classes; pipeline sets both)
      cache swaps:                    int8_kv_cache, ring_kv_cache
      source transforms:              vulkan_sdpa_transform, vulkan_rope_transform
    """

    # --- class-level on Gemma4Attention ---
    use_bshd_kv: bool = True
    use_dynamic_kv_slice: bool = False
    use_bool_mask: bool = False
    use_jit_mask: bool = False
    use_shared_ring_state: bool = False
    skip_redundant_rope_slice: bool = False
    use_prebroadcast_rope: bool = False

    # --- instance-level, set per Gemma4Attention after construction ---
    use_broadcast_gqa: bool = False
    use_custom_sdpa: bool = False

    # --- class-level on RMSNorm (both text and audio) ---
    use_rsqrt: bool = True

    # --- KV cache swaps (XNNPACK only; the Vulkan SDPA transform rejects both) ---
    int8_kv_cache: bool = False
    ring_kv_cache: bool = False

    # --- Vulkan source transforms ---
    vulkan_sdpa_transform: bool = False
    vulkan_rope_transform: bool = False

    # --- output shaping ---
    return_last_only: bool = True


@dataclass(frozen=True)
class Recipe:
    """One complete, runnable export configuration."""

    name: str
    backend: Backend
    modality: Modality

    quant: QuantSpec = field(default_factory=QuantSpec)
    lengths: LengthSpec = field(default_factory=LengthSpec)
    attn: AttnFlags = field(default_factory=AttnFlags)

    # The audio tower runs on XNNPACK even inside an otherwise-Vulkan PTE:
    # the Vulkan gemma_custom_sdpa per-head+softcap shader measured cos 0.40.
    audio_backend: Backend = "xnnpack"

    # Provenance and trust.
    source_script: str = ""
    validated_on_device: bool = False
    notes: str = ""

    # ---- token ids, fixed by the local Gemma4 tokenizer fork ----
    bos_id: int = 2
    eos_ids: tuple[int, ...] = (1, 106, 50)
    vocab_size: int = 262144
    image_placeholder_id: int = 258880
    audio_placeholder_id: int = 258881

    @property
    def has_vision(self) -> bool:
        return self.modality == "multimodal"

    @property
    def has_audio(self) -> bool:
        return self.modality == "multimodal"

    @property
    def method_names(self) -> tuple[str, ...]:
        if self.modality == "text":
            # A single-method PTE MUST name its method "forward": the runner's
            # text_decoder_runner.h hardcodes load_method("forward").
            return ("forward",)
        return ("token_embedding", "text_decoder", "vision_encoder", "audio_encoder")

    def overridden(self, **kwargs) -> "Recipe":
        """Return a copy with top-level fields replaced.

        Nested specs are replaced wholesale; use `replace_nested` for a single
        nested field.
        """
        return dataclasses.replace(self, **kwargs)

    def replace_nested(self, section: str, **kwargs) -> "Recipe":
        """Return a copy with fields of one nested spec replaced.

        e.g. `recipe.replace_nested("lengths", max_seq_len=4096)`
        """
        current = getattr(self, section)
        return dataclasses.replace(self, **{section: dataclasses.replace(current, **kwargs)})

    def constant_methods(self) -> dict:
        """The metadata dict baked into the PTE.

        `get_max_seq_len` is read by the runner as the PER-CALL PREFILL CHUNK
        SIZE, not as the context limit — `get_max_context_len` is the context
        limit. Setting `get_max_seq_len` to the graph's prefill bound is
        deliberate here: it makes the runner chunk at exactly the bound the
        graph accepts. The cost is that prompts are capped per call at that
        value. See docs/RECIPES.md.
        """
        cm = {
            "enable_dynamic_shape": True,
            "use_kv_cache": True,
            "get_max_seq_len": self.lengths.effective_text_decoder_prefill_len,
            "get_max_context_len": self.lengths.max_seq_len,
            "get_vocab_size": self.vocab_size,
            "get_bos_id": self.bos_id,
            # list, not tuple — the RN-ET runner expects a list here.
            "get_eos_ids": list(self.eos_ids),
        }
        if self.modality == "multimodal":
            cm.update(
                {
                    "has_ple": True,
                    "image_placeholder_id": self.image_placeholder_id,
                    "audio_placeholder_id": self.audio_placeholder_id,
                    "audio_soft_tokens_max": self.lengths.audio_soft_tokens_max,
                    "audio_block_samples": self.lengths.audio_block_samples,
                    "audio_block_k_min": self.lengths.audio_k_min,
                    "audio_block_k_max": self.lengths.audio_k_max,
                    "audio_sample_count_min": self.lengths.audio_sample_count_min,
                    "audio_sample_count_max": self.lengths.audio_sample_count_max,
                    "audio_sample_rate": self.lengths.audio_sample_rate,
                }
            )
        return cm

    def summary(self) -> str:
        """Human-readable one-screen description, printed by the CLI."""
        L, Q, A = self.lengths, self.quant, self.attn
        tds = L.effective_text_decoder_prefill_len
        lines = [
            f"recipe            {self.name}   ({self.backend}, {self.modality})",
            f"source            {self.source_script or 'n/a'}",
            f"validated on dev  {self.validated_on_device}",
            f"methods           {', '.join(self.method_names)}",
            "",
            f"layers            {L.num_layers} ({L.num_kv_shared_layers} kv-shared)",
            f"max_seq_len       {L.max_seq_len}   (KV cache depth)",
            f"prefill_len       {L.prefill_len}   (token-embedding / forward seq bound)",
            f"text_dec_prefill  {tds}   (text_decoder seq bound; sizes the arena)",
            "",
            f"linear quant      int{Q.linear_bits} / g{Q.linear_group_size}"
            + (f"  skip={list(Q.skip_linear_name_suffixes)}" if Q.skip_linear_name_suffixes else ""),
            f"main embedding    int4 / g{Q.embedding_group_size}" if Q.quantize_main_embeddings else "main embedding    unquantized",
            f"PLE embedding     int{Q.ple_embedding_bits} / g{Q.ple_embedding_group_size}" if Q.quantize_ple_embeddings else "PLE embedding     unquantized",
            f"base dtype        {Q.base_dtype}   fp16_retarget={Q.fp16_retarget}",
            f"logits dtype      {Q.logits_dtype}",
        ]
        if self.modality == "multimodal":
            lines += [
                "",
                f"vision            {L.vision_h}x{L.vision_w}  pool_k={L.pool_k}  "
                f"{L.max_soft_tokens} soft tokens  quant={Q.vision_quant}",
                f"audio             {L.audio_block_samples}*k, k in [{L.audio_k_min},{L.audio_k_max}]  "
                f"T_soft<={L.audio_soft_tokens_max}  quant={Q.audio_quant}  backend={self.audio_backend}",
            ]
        lines += [
            "",
            "attn flags        " + ", ".join(
                n for n, v in dataclasses.asdict(A).items() if v is True
            ),
        ]
        if self.notes:
            lines += ["", f"notes             {self.notes}"]
        return "\n".join(lines)
