"""Where the export fetches Echo's weights and its audio codec from.

Both come from Hugging Face at pinned revisions. software-mansion/Echo is
private: the download relies on the cached `huggingface-cli login` (or
HF_TOKEN), never on a token stored here.

ECHO_LOCAL_DIR points the export at a local copy laid out like the source repo
(`pretrain/<checkpoint>/{config.json,model.safetensors}`), with the codec in
`codec/`, instead of downloading.
"""

import os
from pathlib import Path

SOURCE_REPO = "software-mansion/Echo"
SOURCE_REVISION = "b6ac38abf3a732d3e5190b3e3722a3235a5ef69f"
# Higgs Audio v2 tokenizer (8 codebooks x 1024 codes, 25 frames/s, 24 kHz).
CODEC_REPO = "eustlb/higgs-audio-v2-tokenizer"
CODEC_REVISION = "528e871c2a26c4f0f7773b9754e2e1acae20899d"

LOCAL_DIR_ENV = "ECHO_LOCAL_DIR"


def _local() -> Path | None:
    local = os.environ.get(LOCAL_DIR_ENV)
    return Path(local).expanduser() if local else None


def checkpoint(name: str) -> str:
    """Directory holding config.json + model.safetensors of `pretrain/<name>`
    ("fp32", "w4a16", "w8a16" or "w8a8")."""
    local = _local()
    if local:
        return str(local / "pretrain" / name)
    from huggingface_hub import snapshot_download

    root = snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION, allow_patterns=[f"pretrain/{name}/*"])
    return str(Path(root) / "pretrain" / name)


def codec() -> str:
    """Directory of the codec checkpoint (transformers HiggsAudioV2TokenizerModel)."""
    local = _local()
    if local:
        return str(local / "codec")
    from huggingface_hub import snapshot_download

    return snapshot_download(
        CODEC_REPO,
        revision=CODEC_REVISION,
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
    )


def source_files(patterns: list[str]) -> str:
    """Root of a SOURCE_REPO snapshot holding `patterns` (tokenizer, samples)."""
    local = _local()
    if local:
        return str(local)
    from huggingface_hub import snapshot_download

    return snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION, allow_patterns=patterns)
