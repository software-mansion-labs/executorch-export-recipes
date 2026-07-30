"""`nn.Module` wrappers that define each exported PTE method's ABI.

Every class here exists for exactly one reason: `torch.export` traces a
module's `forward`, so the *parameter names*, *shapes*, *dtypes* and *output
arity* of these wrappers ARE the PTE method signature. The `dynamic_shapes=`
dict passed to `torch.export.export` is keyed by the parameter names below —
rename a parameter here and the export call must be updated in lockstep.

Provenance (all paths relative to `gemma_export/experiments_vulkan/op_bisect/`):

  ================  ==========================================================
  Class             Source
  ================  ==========================================================
  TokenEmbeddingE.  iter215_mm_vulkan_tds768_export.py:460-468 (`TextEncoder`)
                    == iter211_mm_xnnpack_export.py:719 which reuses
                    `Gemma4TokenEmbedderExport` (gemma4/model/text.py:1454-1464)
  TextDecoderExport iter215_mm_vulkan_tds768_export.py:471-479 (`TextDecoder`)
                    == iter211_mm_xnnpack_export.py:720 which reuses
                    `Gemma4TextDecoderExport` (gemma4/model/text.py:1467-1480)
  VisionEncoderE.   iter215_mm_vulkan_tds768_export.py:482-595 (`VisionEncoder`)
                    + iter211_mm_xnnpack_export.py:392-459
                    (`VisionEncoderRawImageExp93`) — genuinely different, see
                    the class docstring for the flags that select between them
  AudioEncoderE.    iter211_mm_xnnpack_export.py:462-481
                    == iter215_mm_vulkan_tds768_export.py:640-657
  AudioEncoderWavE. iter211_mm_xnnpack_export.py:484-504
                    + iter215_mm_vulkan_tds768_export.py:660-682 — differ ONLY
                    in the output dtype cast; see `output_dtype`
  AudioEncoderEdge  iter215_mm_vulkan_tds768_export.py:602-625 (`AudioEncoder`)
  TowerExport       — the iter213 edge-tower path, UNUSED by iter214/iter215
                    but still dispatched on by `_export_4method` (iter215:1443)
  SingleMethodFwdE. iter212_text_vulkan_export.py:83-171
                    (`_ProductionForwardKVShared`)
                    + iter210_text_xnnpack_export.py:245-253, which exports the
                    bare `Gemma4EdgeForCausalLM` with no wrapper at all
  ================  ==========================================================

Method-name mapping (react-native-executorch `constants.h`):

  token_embedding  kTokenEmbeddingMethod  TokenEmbeddingExport
  text_decoder     kTextModelMethod       TextDecoderExport
  vision_encoder   kVisionEncoderMethod   VisionEncoderExport
  audio_encoder    kAudioEncoderMethod    AudioEncoderWavExport
  forward          (text-only PTEs)       SingleMethodForwardExport

Text-only PTEs deliberately do NOT split into token_embedding + text_decoder:
that split exists solely to expose a vision/audio splice point (iter210:1-14),
and `text_decoder_runner.h` hardcodes `load_method("forward")`.

Concrete dims for the shipped Gemma4 E2B config (`Gemma4EdgeConfig`
defaults): hidden_size=1536, ple_dim=256, L=num_hidden_layers=35,
vocab_size=262144.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from gemma4.model.audio import Gemma4EdgeAudioTower
from gemma4.model.mel_frontend import MelFrontend
from gemma4.model.text import Gemma4EdgeForCausalLM
from gemma4.model.vision import Gemma4EdgeVisionTower

__all__ = [
    "TokenEmbeddingExport",
    "TextDecoderExport",
    "VisionEncoderExport",
    "AudioEncoderExport",
    "AudioEncoderWavExport",
    "AudioEncoderEdgeTowerExport",
    "SingleMethodForwardExport",
]


# ---------------------------------------------------------------------------
# Method "token_embedding"
# ---------------------------------------------------------------------------

class TokenEmbeddingExport(nn.Module):
    """PTE method ``token_embedding`` — word embeddings + per-layer PLE tokens.

    Thin shim over `Gemma4EdgeForCausalLM.embed_for_export`, which requires
    `config.use_ple=True` and a prior `model.model.split_ple_for_export()`
    call. Used verbatim by BOTH multimodal recipes (iter211 XNNPACK reuses
    `gemma4.model.text.Gemma4TokenEmbedderExport`; iter215 Vulkan declares an
    identical local class named `TextEncoder` — same body, same ABI).

    Inputs
    ------
    input_ids : int64, ``(1, S)``
        Raw token ids. Traced with ``dynamic_shapes={"input_ids": {1: seq_dim}}``.
        seq_dim max = 1024 (iter211, ``PREFILL_CHUNK_LEN``) / 2048 (iter215,
        ``PREFILL_LEN``).

    Outputs
    -------
    inputs_embeds : model dtype (fp16 in both shipping recipes), ``(1, S, hidden_size)``
        The runner splices vision/audio encoder outputs into THIS tensor at the
        placeholder positions.
    ple_tokens : model dtype, ``(1, S, L, ple_dim)``
        Per-layer PLE token signal keyed on the RAW input_ids. Gemma4's PLE
        token path fires at the placeholder slot regardless of whether the
        final embedding ends up being text, vision or audio — so the caller
        splices encoder outputs into `inputs_embeds` only and leaves
        `ple_tokens` untouched.

    The 2-tuple output is why multimodal PTEs bake ``"has_ple": True`` into
    `constant_methods`: the RNE runner unpacks `token_embedding.execute()` as
    a 2-tuple (kHasPLE in constants.h) instead of a single tensor.
    """

    def __init__(self, lm: Gemma4EdgeForCausalLM) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lm.embed_for_export(input_ids)


# ---------------------------------------------------------------------------
# Method "text_decoder"
# ---------------------------------------------------------------------------

class TextDecoderExport(nn.Module):
    """PTE method ``text_decoder`` — spliced embeddings -> logits. Owns the KV cache.

    Thin shim over `Gemma4EdgeForCausalLM.decode_from_embeds`.

    NAMING CONFLICT (reconciled): iter211 reuses
    `gemma4.model.text.Gemma4TextDecoderExport`, whose parameters are named
    ``(inputs_embeds, ple_tok, input_pos)``; iter215's local `TextDecoder`
    names them ``(embeds, ple_tokens, input_pos)``. The bodies are identical.
    This class uses the iter215 names. Because `dynamic_shapes` is keyed by
    parameter name, an export driver using THIS class must pass::

        dynamic_shapes={"embeds":     {1: seq_dim},
                        "ple_tokens": {1: seq_dim},
                        "input_pos":  {0: seq_dim}}

    and NOT iter211's ``{"inputs_embeds": ..., "ple_tok": ...}``.

    Inputs
    ------
    embeds : model dtype (fp16), ``(1, S, hidden_size)``
        `token_embedding`'s first output, with vision/audio soft tokens already
        spliced in at the placeholder positions by the runner.
    ple_tokens : model dtype (fp16), ``(1, S, L, ple_dim)``
        `token_embedding`'s second output, passed through unmodified.
    input_pos : int64, ``(S,)``
        Absolute cache positions for this chunk. Drives the RoPE gather and the
        KV-cache write offsets.

    Outputs
    -------
    logits : ``logits_dtype`` (fp16 in both recipes), ``(1, S_out, vocab_size)``
        ``S_out == 1`` when ``lm.return_last_only=True`` (iter211's multimodal
        setting), else ``S``.

    Sizing note: `text_decoder`'s seq_dim is bounded INDEPENDENTLY of
    `token_embedding`'s. iter215 caps it at ``TEXT_DECODER_PREFILL_LEN=768``
    (>= the 744 audio soft tokens at k=62) while `token_embedding` runs to
    ``PREFILL_LEN=2048``; iter214's 128 cap silently produced whitespace-only
    output for vision (280 tokens) and audio (744 tokens) because
    `multimodal_prefiller.cpp` hands the encoder output's full seq dim to
    `text_decoder.execute()` in one shot, without chunking.
    """

    def __init__(self, lm: Gemma4EdgeForCausalLM) -> None:
        super().__init__()
        self.lm = lm

    def forward(
        self,
        embeds: torch.Tensor,
        ple_tokens: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm.decode_from_embeds(embeds, ple_tokens, input_pos)


# ---------------------------------------------------------------------------
# Method "vision_encoder"
# ---------------------------------------------------------------------------

class VisionEncoderExport(nn.Module):
    """PTE method ``vision_encoder`` — raw NCHW image -> soft tokens. Fully static.

    Inputs
    ------
    pixel_values : fp32, ``(1, 3, VIS_H, VIS_W)`` — STATIC, no dynamic dims
        Raw 0..255 RGB (NOT pre-normalised): the ``* (1/255)`` rescale and the
        patchify unfold both live inside the graph. ``VIS_H = vis_ny *
        patch_size``, ``VIS_W = vis_nx * patch_size``; both recipes ship
        448x640 (28x40 patches, POOL_K=2, 280 soft tokens) — 448x640 is the
        caption-coherence floor, 224x320 hallucinates.
        The runner MUST pass a contiguous tensor: ExecuTorch reads the raw
        buffer against the PTE's expected strides, so a non-contiguous view
        silently yields cos~=0.7 rather than an error.

    Outputs
    -------
    embeds : ``(1, max_soft_tokens, hidden_size)``
        Vision soft tokens, already projected through `embed_vision` into the
        text decoder's embedding space.
    ple_tokens : ``(1, max_soft_tokens, L, ple_dim)`` — ONLY when
        ``emit_ple_tokens=True``; the forward then returns a 2-tuple.

    ---------------------------------------------------------------------
    LINEAGE CONFLICT — the two sources differ substantially; both are kept.
    ---------------------------------------------------------------------
    iter211 (`VisionEncoderRawImageExp93`, XNNPACK, wraps the *HF*
    `Gemma4VisionEncoder` after monkey-patching its attention onto
    `llama.custom_sdpa`):
        encoder_style="layer_loop", emit_ple_tokens=False,
        expand_position_buffers=False, mask_padding_before_pool=False
    iter215 (`VisionEncoder`, Vulkan, wraps `Gemma4EdgeVisionTower` with
    B1/B2/B3/B5 flags on):
        encoder_style="module", emit_ple_tokens=True,
        expand_position_buffers=True, mask_padding_before_pool=True

    `encoder_style` is a real ABI-level fork:
      - "module"     calls ``self.encoder(inputs_embeds=..., attention_mask=~padding,
                     pixel_position_ids=...)`` and reads ``.last_hidden_state``.
                     Required by `Gemma4EdgeVisionEncoder`, which owns the
                     static-mask bake (B3) and the Vulkan SDPA path (B5).
      - "layer_loop" drives ``encoder.rotary_emb`` + ``encoder.layers`` by hand
                     with ``attention_mask=None``. This is what iter211 needs
                     because its per-layer forward was replaced wholesale by
                     `_apply_vision_custom_sdpa_patch`, which ignores
                     attention_mask and hardcodes is_causal=False.

    `expand_position_buffers` and `mask_padding_before_pool` are NOT maths
    changes at the PTE contract (B is statically 1 and `padding_positions` is
    all-False — the contract admits no padded patches). They are graph-shape
    choices: iter211 stores B=1 buffers and skips ``.expand`` so no runtime
    `aten.expand_copy` is emitted; the same reasoning killed 35x expand_copy
    on the Vulkan side (iter215 Fix D). Left independently settable so neither
    source's exact graph is lost. Defaults follow `encoder_style`.

    `emit_ple_tokens=True` bakes ``ple_layer(pad_ids_const)`` per layer as a
    persistent constant buffer (iter215 Fix C). For a CONSTANT input
    (pad_token_id repeated max_soft_tokens times) that output is itself a
    constant ``(1, max_soft_tokens, ple_dim)`` tensor, so baking it at
    construction time drops 35x `aten.embedding` + 35x
    `torchao.dequantize_affine` from the vision_encoder graph and shifts the
    int2 PLE weight tables to the text_decoder method only.
    """

    def __init__(
        self,
        vision_tower: Union[Gemma4EdgeVisionTower, nn.Module],
        embed_vision: nn.Module,
        *,
        vis_nx: int,
        vis_ny: int,
        patch_size: int = 16,
        pool_k: int = 2,
        max_soft_tokens: Optional[int] = None,
        encoder_style: str = "module",
        emit_ple_tokens: bool = False,
        lm: Optional[Gemma4EdgeForCausalLM] = None,
        pad_token_id: int = 0,
        expand_position_buffers: Optional[bool] = None,
        mask_padding_before_pool: Optional[bool] = None,
        pixel_position_ids: Optional[torch.Tensor] = None,
        padding_positions: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if encoder_style not in ("module", "layer_loop"):
            raise ValueError(
                f"encoder_style must be 'module' or 'layer_loop', got {encoder_style!r}"
            )
        self.encoder_style = encoder_style

        # Lineage-derived defaults; see the class docstring. Both are no-ops
        # mathematically at B=1 with all-False padding — they only change how
        # many host ops the exported graph carries.
        if expand_position_buffers is None:
            expand_position_buffers = encoder_style == "module"
        if mask_padding_before_pool is None:
            mask_padding_before_pool = encoder_style == "module"
        self.expand_position_buffers = bool(expand_position_buffers)
        self.mask_padding_before_pool = bool(mask_padding_before_pool)

        self.patch_embedder = vision_tower.patch_embedder
        self.encoder = vision_tower.encoder
        self.pooler = vision_tower.pooler
        self.config = vision_tower.config
        self.embed_vision = embed_vision

        self.standardize = getattr(self.config, "standardize", False)
        if self.standardize:
            self.register_buffer("std_bias", vision_tower.std_bias.detach().clone())
            self.register_buffer("std_scale", vision_tower.std_scale.detach().clone())

        self.patch_size = int(patch_size)
        self.vis_nx = int(vis_nx)
        self.vis_ny = int(vis_ny)
        self.pool_k = int(pool_k)
        max_patches = self.vis_nx * self.vis_ny
        if max_soft_tokens is None:
            max_soft_tokens = max_patches // (self.pool_k * self.pool_k)
        self.max_patches = max_patches
        self.max_soft_tokens = int(max_soft_tokens)
        assert self.max_patches == self.max_soft_tokens * self.pool_k * self.pool_k
        assert self.vis_nx % self.pool_k == 0 and self.vis_ny % self.pool_k == 0

        # Canonical row-major (gx, gy) patch grid. -1 would mark padding; the
        # PTE contract has none, so `padding_positions` is all-False.
        if pixel_position_ids is None:
            gx = torch.arange(self.vis_nx).repeat(self.vis_ny)
            gy = torch.arange(self.vis_ny).repeat_interleave(self.vis_nx)
            pixel_position_ids = torch.stack([gx, gy], dim=-1).unsqueeze(0).long()
        if padding_positions is None:
            padding_positions = torch.zeros(1, self.max_patches, dtype=torch.bool)
        assert pixel_position_ids.shape[0] == 1, (
            "B=1 pos_ids required; .expand(1,-1,-1) is a no-op but the stored "
            "buffer must already be (1, P, 2)."
        )
        assert padding_positions.shape[0] == 1, (
            "B=1 padding required; stored buffer must be (1, P)."
        )
        self.register_buffer(
            "pixel_position_ids", pixel_position_ids.detach().clone(), persistent=False
        )
        self.register_buffer(
            "padding_positions", padding_positions.detach().clone(), persistent=False
        )

        self.emit_ple_tokens = bool(emit_ple_tokens)
        self.pad_token_id = int(pad_token_id)
        self._num_ple_layers = 0
        if self.emit_ple_tokens:
            if lm is None:
                raise ValueError("emit_ple_tokens=True requires `lm=` to read "
                                 "lm.model._ple_per_layer_embs")
            # Fix C: bake PLE per-layer pad-token embeddings as constants.
            # The output of `ple_layer(pad_ids_const)` for a CONSTANT input
            # (pad_token_id repeated max_soft_tokens times) is itself a
            # constant `(1, max_soft_tokens, ple_dim)` tensor. By baking it
            # at construction time, the exported graph for vision_encoder
            # drops 35x aten.embedding + 35x torchao.dequantize_affine and
            # the int2 PLE weight tables shift to text_decoder-only.
            self._num_ple_layers = len(lm.model._ple_per_layer_embs)
            pad_ids_const = torch.full(
                (1, self.max_soft_tokens), self.pad_token_id, dtype=torch.long,
            )
            with torch.no_grad():
                for i, ple_layer in enumerate(lm.model._ple_per_layer_embs):
                    emb = ple_layer(pad_ids_const).detach().contiguous()
                    # Persistent so the constant lives inside the
                    # vision_encoder method (don't depend on the LM module's
                    # PLE tables).
                    self.register_buffer(f"_baked_ple_pad_{i}", emb, persistent=True)

    def _soft_tokens(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        image = image * (1.0 / 255.0)

        # Patchify MUST be channels-LAST: HF expects (ny, nx, p, p, C), so the
        # permute is (0, 2, 3, 4, 5, 1). The planar-RGB (0, 2, 3, 1, 4, 5)
        # form used by exp89/exp90 dropped eager parity to cos=0.366.
        p = self.patch_size
        image = image.unfold(2, p, p).unfold(3, p, p)
        image = image.permute(0, 2, 3, 4, 5, 1).contiguous()
        pixel_values = image.flatten(3).flatten(1, 2)

        if self.expand_position_buffers:
            pos_ids = self.pixel_position_ids.expand(B, -1, -1)
            padding = self.padding_positions.expand(B, -1)
        else:
            # iter211 patch (2): no .expand — use the B=1 buffers directly so
            # the graph carries no runtime aten.expand_copy.
            pos_ids = self.pixel_position_ids
            padding = self.padding_positions

        inputs_embeds = self.patch_embedder(pixel_values, pos_ids, padding)

        if self.encoder_style == "module":
            enc_out = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=~padding,
                pixel_position_ids=pos_ids,
            )
            hidden = enc_out.last_hidden_state
        else:
            # iter211 layer-loop: the per-layer forward was replaced by
            # `_apply_vision_custom_sdpa_patch` (llama.custom_sdpa, BSHD, no
            # GQA, attention_mask=None, is_causal=False, start_pos=0), so the
            # encoder module's mask plumbing is bypassed entirely.
            hidden = inputs_embeds
            position_embeddings = self.encoder.rotary_emb(hidden, pos_ids)
            for layer in self.encoder.layers[: self.encoder.config.num_hidden_layers]:
                hidden = layer(
                    hidden,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                    position_ids=pos_ids,
                )

        if self.mask_padding_before_pool:
            hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)

        pooled, _ = self.pooler._avg_pool_by_positions(
            hidden, pos_ids, self.max_soft_tokens
        )
        pooled = pooled * self.pooler.root_hidden_size
        if self.standardize:
            pooled = (pooled - self.std_bias) * self.std_scale

        return self.embed_vision(pooled)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embeds = self._soft_tokens(pixel_values)
        if not self.emit_ple_tokens:
            return embeds
        # Fix C + Fix D: use the baked PLE pad-token embeddings (constants)
        # WITHOUT an expand. B is statically 1 in vision_encoder, so
        # `expand(B, -1, -1)` was a no-op that emitted 35x runtime
        # aten.expand_copy ops on the host. Stack the buffers directly; each
        # is shape (1, max_soft_tokens, ple_dim) and stacks on dim=2 to give
        # (1, max_soft_tokens, N, ple_dim) — bit-exact match to the pre-Fix-D
        # output (which had B=1).
        per_layer = [
            getattr(self, f"_baked_ple_pad_{i}") for i in range(self._num_ple_layers)
        ]
        ple_tokens = torch.stack(per_layer, dim=2)
        return embeds, ple_tokens


# ---------------------------------------------------------------------------
# Method "audio_encoder"
# ---------------------------------------------------------------------------

class AudioEncoderExport(nn.Module):
    """Inner mel-features-in audio encoder. NOT exported directly.

    Composed by `AudioEncoderWavExport`; kept as its own module because both
    sources keep it as one (iter211 `AudioEncoderExport`, iter215
    `_Iter211AudioEncoderExport`) and the export driver's parity harness runs
    it standalone against HF `get_audio_features`.

    Wraps the STOCK HF `Gemma4AudioTower` (not the edge re-implementation) plus
    HF `embed_audio`.

    DTYPE HISTORY — do not "optimise" this back to fp16 weights: iter211's
    original docstring claimed the tower was fp16-internal via `.half()` at
    build time, but its `main()` explicitly keeps the audio encoder fp32 and
    the code performs no cast. `diag_pte_decode_with_hf_audio.py` (2026-04-23)
    found the fp16 PTE hit min_cos=0.85 on a row and flipped the first-token
    argmax 818 -> 100, despite eager fp16 min_cos=0.99996. Weights & buffers
    stay fp32; only the *output* may be cast (see `AudioEncoderWavExport`).

    Inputs
    ------
    input_features : fp32, ``(1, T, 128)`` log-mel, T = 48*k
    attention_mask : bool, ``(1, T)``

    Outputs
    -------
    audio_embeds : fp32, ``(1, T_soft, hidden_size)`` (T_soft = 12*k)
    """

    def __init__(self, audio_tower: nn.Module, embed_audio: nn.Module) -> None:
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio

    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        audio_out = self.audio_tower(input_features, attention_mask=attention_mask)
        hidden = audio_out.last_hidden_state
        return self.embed_audio(hidden)


class AudioEncoderWavExport(nn.Module):
    """PTE method ``audio_encoder`` — raw fp32 mono 16 kHz PCM -> audio embeds.

    Contract: input shape ``[1, N_samples]`` where ``N_samples = 7680 * k``
    (``BLOCK_SAMPLES * k``). `MelFrontend` produces ``T_raw = 48*k - 1`` mel
    frames for that length; we right-pad by exactly 1 frame (zeros, masked
    off) so the audio_tower sees ``T = 48*k``, matching the chunked-attention
    block size.

    Inputs
    ------
    waveform : fp32, ``(1, 7680*k)``
        Traced with a MULTIPLE-of-block dynamic dim::

            k_dim = Dim("audio_k", min=1, max=62)
            dynamic_shapes = ({1: 7680 * k_dim}, None)

        iter211 spells the same thing as ``Dim("audio_blocks", min=1, max=63)``
        with ``AUDIO_SAMPLES_PER_BLOCK=7680``; iter215 uses ``BLOCK_K_MAX=62``
        from `gemma4.model.audio`. See the k_max note in the report.
    num_valid_samples : int64 0-d scalar tensor
        Real PCM length before the caller's zero-padding to block alignment.
        Frames whose 20 ms analysis window extends past this are masked out.
        Declared ``None`` in dynamic_shapes (scalar, no dyn dim).

    Outputs
    -------
    audio_embeds : ``(1, 12*k, hidden_size)`` — dtype set by `output_dtype`
        ``T_soft = 12*k``; 744 at k=62.

    ---------------------------------------------------------------------
    LINEAGE CONFLICT — output dtype only. Everything else is byte-identical.
    ---------------------------------------------------------------------
    iter211 (XNNPACK) returns the encoder output unchanged: fp32.
    iter215 (Vulkan) appends ``out.to(torch.float16)`` so the host splice into
    the text_decoder's fp16 `embeds_buf` is a clean memcpy.
    Pass ``output_dtype=None`` for the XNNPACK lineage (no cast at all — NOT
    "cast to fp32", so an fp64/bf16 tower would pass its dtype straight
    through) and ``output_dtype=torch.float16`` for the Vulkan lineage.
    """

    def __init__(
        self,
        audio_tower: nn.Module,
        embed_audio: nn.Module,
        *,
        output_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.mel_frontend = MelFrontend()
        self.inner = AudioEncoderExport(audio_tower, embed_audio)
        self.output_dtype = output_dtype

    def forward(
        self, waveform: torch.Tensor, num_valid_samples: torch.Tensor
    ) -> torch.Tensor:
        log_mel, mask = self.mel_frontend(waveform, num_valid_samples)
        # MelFrontend emits T_raw = 48*k - 1; pad +1 zero frame so the
        # audio_tower's chunked attention sees T = 48*k.
        log_mel = F.pad(log_mel, (0, 0, 0, 1))
        mask = F.pad(mask, (0, 1), value=False)
        out = self.inner(log_mel, mask)
        if self.output_dtype is not None:
            # Cast so the host splice into the text_decoder's fp16 embeds_buf
            # is a clean memcpy (Vulkan lineage convention).
            out = out.to(self.output_dtype)
        return out


class AudioEncoderEdgeTowerExport(nn.Module):
    """PTE method ``audio_encoder``, EDGE-TOWER variant. Not used by any
    shipping recipe — retained because `_export_4method` still dispatches on it.

    iter213 wrapped `Gemma4EdgeAudioTower` (baked softplus / pre-baked 5D mask
    / rel_shift index_select) instead of the stock HF tower. iter214 swapped
    it back out for `AudioEncoderWavExport` after iter213 produced
    uniform-noise logits on every modality; iter215 keeps this class only so a
    side-by-side diff with iter213 stays readable. The edge tower carries its
    own log-mel frontend, so there is no `MelFrontend` here — the wrapper
    builds the sample-level mask itself and hands the raw waveform over.

    Same external ABI as `AudioEncoderWavExport`:
      waveform fp32 ``(1, 7680*k)``, num_valid_samples int64 0-d
      -> audio_embeds fp16 ``(1, 12*k, hidden_size)``.
    """

    def __init__(
        self,
        edge_tower: Gemma4EdgeAudioTower,
        *,
        output_dtype: Optional[torch.dtype] = torch.float16,
    ) -> None:
        super().__init__()
        self.edge_tower = edge_tower
        self.output_dtype = output_dtype

    def forward(
        self, waveform: torch.Tensor, num_valid_samples: torch.Tensor
    ) -> torch.Tensor:
        n = waveform.shape[1]
        attention_mask = (
            torch.arange(n, device=waveform.device) < num_valid_samples
        ).unsqueeze(0)
        out = self.edge_tower(waveform, attention_mask)
        if self.output_dtype is not None:
            out = out.to(self.output_dtype)
        return out


# ---------------------------------------------------------------------------
# Method "forward" (text-only PTEs)
# ---------------------------------------------------------------------------

class SingleMethodForwardExport(nn.Module):
    """PTE method ``forward`` — the ONE method of the two text-only recipes.

    The method MUST be named "forward": `text_decoder_runner.h` hardcodes
    ``load_method("forward")`` and a PTE exposing only e.g. `forward_decode`
    SIGABRTs at load.

    Inputs
    ------
    input_ids : int64, ``(1, S)``
    input_pos : int64, ``(S,)``
        Traced with ``dynamic_shapes={"input_ids": {1: seq_dim},
        "input_pos": {0: seq_dim}}``; ``Dim("seq_len", min=1, max=...)`` with
        min=1 so the single method serves decode (S=1) as well as prefill.
        max = 1024 (iter210, ``PREFILL_CHUNK_LEN``) / 128 (iter212,
        ``PREFILL_LEN``; runners must chunk prefill at <= 128 there).

    Outputs
    -------
    logits : ``lm.logits_dtype`` (fp16 in both recipes), ``(1, S_out, vocab_size)``
        ``S_out == 1`` when ``lm.return_last_only=True`` (iter212), else ``S``
        (iter210 sets it False so the runner can read the last-position slice
        after a prefill chunk, matching
        ``logits += (num_tokens - 1) * vocab_size`` in text_decoder_runner.cpp).

    ---------------------------------------------------------------------
    LINEAGE CONFLICT — `use_inline_kv_shared_loop`
    ---------------------------------------------------------------------
    use_inline_kv_shared_loop=False  (iter210, XNNPACK)
        Delegates straight to ``self.lm(input_ids, input_pos)``. iter210 in
        fact exports the bare `Gemma4EdgeForCausalLM` with no wrapper at all;
        this shim preserves both the behaviour and the parameter names, so the
        `dynamic_shapes` dict is unchanged either way.

    use_inline_kv_shared_loop=True   (iter212, Vulkan)
        Re-implements `Gemma4EdgeForCausalLM.forward` INLINE with KV-sharing
        threading — CLEAN, only logits returned, no per-layer taps. Mirrors
        `Gemma4EdgeModel.forward`'s shared_k/shared_v threading with
        sliding_k/v + full_k/v carries; with NUM_KV_SHARED=20 the last 20
        layers are consumers that pull from full_k/v (last global source) or
        sliding_k/v (last sliding source) per their layer_type.

        WHY IT EXISTS: RoPE is gathered with
        ``F.embedding(input_pos, text_model._cos_local)`` etc. rather than
        ``text_model._cos_local[input_pos]``. `F.embedding` lowers to
        `aten.embedding.default` (Vulkan-supported); the subscript form lowers
        to `aten.index.Tensor`, which the Vulkan partitioner rejects — that
        one rejection splits the graph and is what keeps/breaks the whole
        decoder living in a single Vulkan partition.

        CAVEAT (verify before choosing the flag): that F.embedding gather has
        since been folded into `Gemma4EdgeModel.forward` itself
        (gemma4/model/text.py:1086-1089, with the same comment), so the
        False path is ALSO Vulkan-clean for RoPE now. What still differs on
        the True path is that the inline loop:
          * skips the `use_prebroadcast_rope` unsqueeze block,
          * never passes `ring_indices` / `ring_mask` to layers (the Vulkan
            v0 recipe has no ring sliding cache), and
          * assumes `split_ple_for_export()` has been called — it always
            indexes `_ple_per_layer_embs[i]` and has no non-split fallback.
        Note that `decode_from_embeds` (text.py:1263-1266) still uses the
        `[input_pos]` subscript form, so the multimodal `text_decoder` method
        does NOT get the F.embedding treatment.
    """

    def __init__(
        self,
        lm: Gemma4EdgeForCausalLM,
        *,
        use_inline_kv_shared_loop: bool = False,
    ) -> None:
        super().__init__()
        self.lm = lm
        self.use_inline_kv_shared_loop = bool(use_inline_kv_shared_loop)

    def forward(
        self, input_ids: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_inline_kv_shared_loop:
            return self.lm(input_ids, input_pos)

        cfg = self.lm.config
        text_model = self.lm.model

        B, S = input_ids.shape

        hidden = text_model.embed_tokens(input_ids)
        hidden_for_ple = hidden

        # F.embedding lowers to aten.embedding.default (Vulkan-supported);
        # `tensor[input_pos]` lowers to aten.index.Tensor which Vulkan rejects.
        cos_local = F.embedding(input_pos, text_model._cos_local)
        sin_local = F.embedding(input_pos, text_model._sin_local)
        cos_global = F.embedding(input_pos, text_model._cos_global)
        sin_global = F.embedding(input_pos, text_model._sin_global)

        sliding_k: Optional[torch.Tensor] = None
        sliding_v: Optional[torch.Tensor] = None
        full_k: Optional[torch.Tensor] = None
        full_v: Optional[torch.Tensor] = None

        for i, layer in enumerate(text_model.layers):
            is_global = text_model.layer_is_global[i]
            cos = cos_global if is_global else cos_local
            sin = sin_global if is_global else sin_local

            ple_signal = None
            if cfg.use_ple:
                ple_tok_i = text_model._ple_per_layer_embs[i](input_ids)
                ple_ctx_i = text_model._ple_per_layer_projs[i](hidden_for_ple)
                ple_ctx_i = ple_ctx_i * (cfg.hidden_size ** -0.5)
                ple_ctx_i = text_model.per_layer_projection_norm(ple_ctx_i)
                ple_signal = (ple_ctx_i + ple_tok_i) * (2.0 ** -0.5)

            attn = layer.self_attn
            if attn.is_kv_shared_layer:
                if is_global:
                    shared_k, shared_v = full_k, full_v
                else:
                    shared_k, shared_v = sliding_k, sliding_v
            else:
                shared_k, shared_v = None, None

            hidden, k_out, v_out = layer(
                hidden, cos, sin,
                input_pos=input_pos,
                ple_signal=ple_signal,
                shared_k=shared_k, shared_v=shared_v,
            )

            if attn.store_full_length_kv:
                if is_global:
                    full_k, full_v = k_out, v_out
                else:
                    sliding_k, sliding_v = k_out, v_out

        hidden = text_model.norm(hidden)

        if self.lm.return_last_only:
            hidden = hidden[:, -1:, :]

        if self.lm.lm_head is not None:
            logits = self.lm.lm_head(hidden)
        else:
            logits = F.linear(hidden, text_model.embed_tokens.weight)

        if self.lm.logits_dtype != logits.dtype:
            logits = logits.to(self.lm.logits_dtype)

        if cfg.final_logit_softcapping is not None:
            cap = cfg.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)

        return logits
