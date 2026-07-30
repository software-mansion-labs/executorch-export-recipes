"""Pre-flight validation for a Recipe.

Every rule here exists because someone burned real time on it. The point is to
fail in a second, with the measurement or the guard that explains why, instead
of failing thirty minutes into an export — or worse, succeeding and producing a
PTE that emits garbage on device.

Two severities:
  - `Error`   the export cannot work; refuse.
  - `Warning` the export will run, but the result has a known sharp edge.

`check(recipe)` returns (errors, warnings). `assert_valid(recipe)` raises on any
error. The CLI prints warnings and refuses on errors unless `--force`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .config import Recipe


@dataclass(frozen=True)
class Finding:
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


def check(recipe: Recipe) -> Tuple[List[Finding], List[Finding]]:
    errors: List[Finding] = []
    warnings: List[Finding] = []
    L, Q, A = recipe.lengths, recipe.quant, recipe.attn

    # ---------------------------------------------------------------- R1-R3
    # The Vulkan SDPA source transform raises rather than degrading. Catch the
    # same three conditions here, with the guard text, so the failure arrives
    # before the model is built rather than mid-transform.
    if A.vulkan_sdpa_transform:
        if not A.use_bshd_kv:
            errors.append(Finding(
                "R1-vulkan-bshd",
                "vulkan_sdpa_transform requires use_bshd_kv=True. "
                "transforms/sdpa.py raises: 'gemma_sdpa_source_transform requires "
                "Gemma4Attention.use_bshd_kv=True (BSHD layout).'",
            ))
        if A.use_dynamic_kv_slice:
            errors.append(Finding(
                "R2-vulkan-dynslice",
                "use_dynamic_kv_slice is incompatible with the Vulkan SDPA ops. "
                "transforms/sdpa.py raises: 'use_dynamic_kv_slice is not yet "
                "supported by the new Vulkan ops; disable it before applying "
                "gemma_sdpa_source_transform.'",
            ))
        if A.ring_kv_cache:
            errors.append(Finding(
                "R3-vulkan-ring",
                "ring KV cache is incompatible with the Vulkan SDPA ops. "
                "transforms/sdpa.py raises: 'Ring KV cache not supported by "
                "gemma_sdpa_source_transform v0.'",
            ))
        if A.int8_kv_cache:
            # No guard rejects this, which is exactly why it needs flagging:
            # the transform would happily feed int8 cache tensors straight into
            # et_vk::gemma_sdpa_with_kv_cache. Nothing validates that path.
            warnings.append(Finding(
                "R3b-vulkan-int8kv",
                "int8 KV cache under the Vulkan SDPA transform is UNVALIDATED. "
                "No guard rejects it, but no export has ever exercised it: the "
                "int8 cache tensors are passed directly to "
                "et_vk::gemma_sdpa_with_kv_cache with nothing checking dtypes.",
            ))

    # ---------------------------------------------------------------- R4
    # Ring KV depends on the int8 cache swap having run first.
    if A.ring_kv_cache and not A.int8_kv_cache:
        warnings.append(Finding(
            "R4-ring-without-int8",
            "ring_kv_cache without int8_kv_cache: the ring swap is documented to "
            "run AFTER replace_kv_cache_with_quantized_kv_cache so it picks up "
            "the Gemma4QuantizedKVCache instances. Float ring caches work but "
            "are not the configuration any shipped recipe used.",
        ))

    # ---------------------------------------------------------------- R5
    # The iter214 failure, encoded. This is the single most valuable rule here.
    if recipe.modality == "multimodal":
        tds = L.effective_text_decoder_prefill_len
        needed = max(L.max_soft_tokens, L.audio_soft_tokens_max)
        if tds < needed:
            errors.append(Finding(
                "R5-seq-bound",
                f"text_decoder seq bound ({tds}) is below the soft-token count the "
                f"encoders emit (vision {L.max_soft_tokens}, audio "
                f"{L.audio_soft_tokens_max}; need >= {needed}). This is the iter214 "
                "failure exactly: multimodal_prefiller.cpp reads seq_len from the "
                "encoder output and calls execute() in one shot with no chunking "
                "and no bounds check, so exceeding the bound produces silent "
                "garbage (observed: whitespace-only generation), not an error. "
                f"Raise --text-decoder-prefill-len to >= {needed}.",
            ))
        elif tds < needed + 64:
            warnings.append(Finding(
                "R5b-seq-bound-tight",
                f"text_decoder seq bound ({tds}) clears the largest single encoder "
                f"output ({needed}) by only {tds - needed} tokens. Each modality is "
                "a separate one-shot prefill call in "
                "extension/llm/runner/multimodal_prefiller.cpp, so vision and audio "
                "do NOT sum into one call and this is sufficient. The margin still "
                "matters if audio_soft_tokens_max is ever raised (it scales with "
                "audio_k_max), or if a runner batches modalities into one call.",
            ))

    # ---------------------------------------------------------------- R6
    if recipe.modality == "multimodal":
        expected = L.max_soft_tokens * L.pool_k * L.pool_k
        if L.max_patches != expected:
            errors.append(Finding(
                "R6-vision-geometry",
                f"vision geometry inconsistent: {L.vision_h}x{L.vision_w} at "
                f"patch {L.patch_size} gives {L.patch_grid_ny}x{L.patch_grid_nx} = "
                f"{L.max_patches} patches, but max_soft_tokens({L.max_soft_tokens}) "
                f"* pool_k^2({L.pool_k}^2) = {expected}. These must match.",
            ))

    # ---------------------------------------------------------------- R7
    if recipe.modality == "multimodal":
        if L.audio_k_min < 1:
            errors.append(Finding("R7-audio-k", "audio_k_min must be >= 1."))
        if L.audio_k_max < L.audio_k_min:
            errors.append(Finding("R7-audio-k", "audio_k_max must be >= audio_k_min."))
        if L.audio_block_samples != 7680:
            warnings.append(Finding(
                "R7b-audio-block",
                f"audio_block_samples={L.audio_block_samples}, not 7680. The runner "
                "rejects any waveform whose sample count is not an exact multiple "
                "of this value, and 7680 (0.48 s @ 16 kHz) is what every shipped "
                "PTE and the RN-ET runner assume.",
            ))

    # ---------------------------------------------------------------- R8
    # get_max_seq_len semantics. The runner reads it as the per-call prefill
    # chunk size, not the context limit.
    if L.effective_text_decoder_prefill_len < L.max_seq_len:
        warnings.append(Finding(
            "R8-get-max-seq-len",
            f"get_max_seq_len will be baked as {L.effective_text_decoder_prefill_len} "
            f"(the graph's seq bound) while get_max_context_len is {L.max_seq_len}. "
            "This is intentional in the shipped recipes — the runner uses "
            "get_max_seq_len as the per-call prefill chunk size, so it must not "
            "exceed what the graph accepts. The consequence is that a single "
            f"prefill call is capped at {L.effective_text_decoder_prefill_len} "
            "tokens; longer prompts require runner-side chunking.",
        ))

    # ---------------------------------------------------------------- R9
    if recipe.backend == "vulkan" and Q.vision_quant != "none":
        warnings.append(Finding(
            "R9-vulkan-vision-quant",
            "quantising the vision tower on Vulkan re-introduces the "
            "aten.clamp.Tensor problem: Gemma4ClippableLinear's min/max buffers "
            "surface as clamp.Tensor, which the Vulkan op registry does not have "
            "(only clamp.default). The scalar-clamp patch would need to be "
            "restored. iter193b settled on leaving vision fp32.",
        ))

    # ---------------------------------------------------------------- R10
    if recipe.backend == "vulkan" and recipe.audio_backend == "vulkan":
        errors.append(Finding(
            "R10-vulkan-audio",
            "audio on Vulkan is known-broken: the gemma_custom_sdpa per-head + "
            "softcap shader measured cos 0.40 against eager (iter201). Every "
            "recipe since iter202 routes audio to XNNPACK. Set "
            "audio_backend='xnnpack'.",
        ))

    # ---------------------------------------------------------------- R11
    if not A.use_bshd_kv:
        warnings.append(Finding(
            "R11-bshd",
            "use_bshd_kv=False keeps the [B,H,S,D] layout, which costs ~280 extra "
            "transposes. No shipped recipe uses it.",
        ))

    # ---------------------------------------------------------------- R12
    if Q.ple_embedding_bits == 2 and recipe.modality == "multimodal":
        warnings.append(Finding(
            "R12-ple-int2",
            "PLE int2 saves roughly 580 MB but was measured to flip the "
            "multimodal first-token argmax (eager PLE int2 cos 0.895). "
            "iter211 moved to int4 for this reason; the Vulkan multimodal "
            "recipe kept int2. Verify on-device coherence before shipping.",
        ))

    # ---------------------------------------------------------------- R13
    if not recipe.validated_on_device:
        warnings.append(Finding(
            "R13-unvalidated",
            f"recipe '{recipe.name}' has never been validated on device. "
            + (recipe.notes or ""),
        ))

    return errors, warnings


def assert_valid(recipe: Recipe) -> List[Finding]:
    """Raise on any error; return the warnings."""
    errors, warnings = check(recipe)
    if errors:
        detail = "\n  ".join(str(e) for e in errors)
        raise ValueError(
            f"recipe {recipe.name!r} is not valid:\n  {detail}\n\n"
            "Pass --force to export anyway (you almost certainly do not want to)."
        )
    return warnings
