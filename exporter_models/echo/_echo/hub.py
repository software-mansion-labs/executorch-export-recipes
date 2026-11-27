"""Where the export fetches Echo's weights and its audio codec from.

All from public Hugging Face repos at pinned revisions (the v1.0.0 tags that react-native-executorch resolves):

- the model: software-mansion/Reverb-TTS (Echo's release), `model/safetensors/<variant>`: fp32, fp16, and the
  quantized int8 (w8a16), int8a8 (w8a8) and int4 (w4a16, groups of 32) checkpoints, each a config.json (with its
  `quantization` block) and a model.safetensors holding the integer codes and their scales;
- the codec: eustlb/higgs-audio-v2-tokenizer (full precision), and the int8 decoder (GPTQ + QAT of its scales) and
  encoder (QAT) checkpoints from software-mansion/higgs-audio-v2-tokenizer-ExecuTorch `{decoder,encoder}/safetensors/int8`.

REVERB_LOCAL_DIR points the export at a local copy of Reverb-TTS (`model/safetensors/<variant>/`) instead.

The source repo software-mansion/Echo (private, cached `huggingface-cli login`) is only read for its test and dev
sets (evaluate.py, qnn/calibrate.py).
"""

import os
from pathlib import Path

REVERB_REPO = "software-mansion/Reverb-TTS"
REVERB_REVISION = "72be799c8fb6d56a206d260b425c48bf3bd5943d"  # v1.0.0
# Higgs Audio v2 tokenizer (8 codebooks x 1024 codes, 25 frames/s, 24 kHz).
CODEC_REPO = "eustlb/higgs-audio-v2-tokenizer"
CODEC_REVISION = "528e871c2a26c4f0f7773b9754e2e1acae20899d"
CODEC_QUANTIZED_REPO = "software-mansion/higgs-audio-v2-tokenizer-ExecuTorch"
CODEC_QUANTIZED_REVISION = "465d9a2270cc7226599387ace77c34c64b4291be"  # v1.0.0
SOURCE_REPO = "software-mansion/Echo"
SOURCE_REVISION = "b6ac38abf3a732d3e5190b3e3722a3235a5ef69f"

LOCAL_DIR_ENV = "REVERB_LOCAL_DIR"
VARIANTS = ("fp32", "fp16", "int8", "int8a8", "int4")
# The source repo's names for the same checkpoints, which older scripts here still use.
_ALIASES = {"w8a16": "int8", "w8a8": "int8a8", "w4a16": "int4"}


def checkpoint(variant: str) -> str:
    """Directory holding config.json + model.safetensors of the Reverb-TTS checkpoint `variant` (one of VARIANTS;
    "w8a16", "w8a8" and "w4a16" are accepted for int8, int8a8 and int4)."""
    variant = _ALIASES.get(variant, variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown checkpoint {variant!r}, expected one of {VARIANTS}")
    local = os.environ.get(LOCAL_DIR_ENV)
    if local:
        return str(Path(local).expanduser() / "model" / "safetensors" / variant)
    from huggingface_hub import snapshot_download

    root = snapshot_download(REVERB_REPO, revision=REVERB_REVISION, allow_patterns=[f"model/safetensors/{variant}/*"])
    return str(Path(root) / "model" / "safetensors" / variant)


def codec() -> str:
    """Directory of the codec checkpoint (transformers HiggsAudioV2TokenizerModel)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        CODEC_REPO,
        revision=CODEC_REVISION,
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
    )


def _codec_int8(part: str) -> str:
    from huggingface_hub import snapshot_download

    root = snapshot_download(CODEC_QUANTIZED_REPO, revision=CODEC_QUANTIZED_REVISION,
                             allow_patterns=[f"{part}/safetensors/int8/*"])
    return str(Path(root) / part / "safetensors" / "int8")


def decoder_int8() -> str:
    """Directory of the int8 codec decoder: decoder.safetensors (int8 weights, fp16 scales) + quantization.json."""
    return _codec_int8("decoder")


def encoder_int8() -> str:
    """Directory of the int8 QAT codec encoder: encoder.safetensors (int8 weights, fp16 scales) + quantization.json."""
    return _codec_int8("encoder")


def source_files(patterns: list[str]) -> str:
    """Root of a snapshot of the private source repo holding `patterns` (its test and dev sets)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION, allow_patterns=patterns)
