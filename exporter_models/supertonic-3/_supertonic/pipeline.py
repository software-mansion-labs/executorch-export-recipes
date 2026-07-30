"""PyTorch inference pipeline for Supertonic 3.

Replicates the pipeline of the ``supertonic`` package (text preprocessing,
duration prediction, flow-matching latent sampling, vocoding) on top of the
PyTorch models in this package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union
from unicodedata import normalize

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (mirrors supertonic.config)
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = [
    "ar", "bg", "cs", "da", "de", "el", "en", "es", "fi", "fr", "hi", "hr",
    "hu", "id", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ro", "ru",
    "sk", "sv", "sw", "ta", "th", "tl", "tr",
]
UNKNOWN_LANGUAGE = "na"
AVAILABLE_LANGUAGES = SUPPORTED_LANGUAGES + [UNKNOWN_LANGUAGE]

DEFAULT_TOTAL_STEPS = 8
DEFAULT_SPEED = 1.05
DEFAULT_MAX_CHUNK_LENGTH = 300
DEFAULT_MAX_CHUNK_LENGTH_KO = 120
DEFAULT_SILENCE_DURATION = 0.3
MIN_SPEED, MAX_SPEED = 0.7, 2.0

# ---------------------------------------------------------------------------
# Text preprocessing (ported from supertonic.core.UnicodeProcessor)
# ---------------------------------------------------------------------------

_EMOJI_PATTERN = re.compile(
    "[\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "☀-⛿"
    "✀-➿"
    "\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)

_SYMBOL_REPLACEMENTS = {
    "–": "-", "‑": "-", "—": "-", "¯": " ", "_": " ",
    "“": '"', "”": '"', "‘": "'", "’": "'", "´": "'",
    "`": "'", "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
    "→": " ", "←": " ",
}

_SPECIAL_SYMBOLS_PATTERN = re.compile(r"[♥☆♡©\\]")

_PUNCTUATION_SPACING_PATTERNS = [
    (re.compile(r" ,"), ","), (re.compile(r" \."), "."), (re.compile(r" !"), "!"),
    (re.compile(r" \?"), "?"), (re.compile(r" ;"), ";"), (re.compile(r" :"), ":"),
    (re.compile(r" '"), "'"),
]

_DUPLICATE_QUOTES_PATTERN = re.compile(r'(["\'\`])\1+')
_WHITESPACE_PATTERN = re.compile(r"\s+")
_ENDING_PUNCTUATION_PATTERN = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")

_COMMON_ABBREVIATIONS_PATTERN = (
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)"
    r"(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)"
    r"(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)"
    r"(?<=[.!?])\s+"
)


def chunk_text(text: str, max_len: int = DEFAULT_MAX_CHUNK_LENGTH) -> list[str]:
    """Split text into chunks by paragraphs and sentences."""
    if max_len < 10:
        raise ValueError(f"max_len must be at least 10, got {max_len}")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(_COMMON_ABBREVIATIONS_PATTERN, paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 <= max_len:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current and current.strip():
            chunks.append(current.strip())
    return chunks


def length_to_mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    max_len = max_len or lengths.max()
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


class UnicodeProcessor:
    """Text -> unicode index ids + mask (mirrors the original package)."""

    def __init__(self, unicode_indexer_path: Union[str, Path]):
        with open(unicode_indexer_path, "r") as f:
            self.indexer = json.load(f)
        if not isinstance(self.indexer, list) or not self.indexer:
            raise ValueError("Invalid unicode indexer")
        self.supported_chars = {
            chr(u) for u, idx in enumerate(self.indexer) if idx != -1
        }

    def _preprocess_text(self, text: str, lang: Optional[str] = None) -> str:
        try:
            text = normalize("NFKD", text)
        except Exception:
            pass
        text = _EMOJI_PATTERN.sub("", text)
        for old, new in _SYMBOL_REPLACEMENTS.items():
            text = text.replace(old, new)
        text = _SPECIAL_SYMBOLS_PATTERN.sub("", text)
        for k, v in {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}.items():
            text = text.replace(k, v)
        for pattern, replacement in _PUNCTUATION_SPACING_PATTERNS:
            text = pattern.sub(replacement, text)
        text = _DUPLICATE_QUOTES_PATTERN.sub(r"\1", text)
        text = _WHITESPACE_PATTERN.sub(" ", text).strip()
        if not _ENDING_PUNCTUATION_PATTERN.search(text):
            text += "."
        if lang is not None:
            if lang not in AVAILABLE_LANGUAGES:
                raise ValueError(f"Invalid language: '{lang}'")
            text = f"<{lang}>{text}</{lang}>"
        return text

    def validate_text(self, text: str) -> tuple[bool, list[str]]:
        unsupported = set()
        for ch in set(text):
            p_chars = set(self._preprocess_text(ch))
            if p_chars - self.supported_chars:
                unsupported.add(ch)
        return len(unsupported) == 0, sorted(unsupported)

    def __call__(
        self, text_list: list[str], lang: Optional[str] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        texts = [self._preprocess_text(t, lang) for t in text_list]
        lengths = np.array([len(t) for t in texts], dtype=np.int64)
        text_ids = np.zeros((len(texts), lengths.max()), dtype=np.int64)
        for i, text in enumerate(texts):
            ids = [self.indexer[ord(c)] for c in text]
            text_ids[i, : len(ids)] = np.array(ids, dtype=np.int64)
        return text_ids, length_to_mask(lengths)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


class Style:
    """Voice style vectors (ttl: [1, 50, 256], dp: [1, 8, 16])."""

    def __init__(self, style_ttl: np.ndarray, style_dp: np.ndarray):
        self.ttl = np.asarray(style_ttl, dtype=np.float32)
        self.dp = np.asarray(style_dp, dtype=np.float32)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "Style":
        with open(path, "r") as f:
            data = json.load(f)

        def parse(entry):
            return np.array(entry["data"], dtype=np.float32).reshape(*entry["dims"])

        return cls(parse(data["style_ttl"]), parse(data["style_dp"]))


# ---------------------------------------------------------------------------
# TTS engine
# ---------------------------------------------------------------------------


class TTS:
    """PyTorch Supertonic 3 TTS.

    Args:
        model_dir: Directory with ``tts.json``/``unicode_indexer.json`` (the
            repo's ``onnx/`` dir) — used for config + text processing.
        checkpoint: Optional ``.pt`` checkpoint from ``convert.py``. If
            omitted, weights are extracted from the ONNX files on the fly.
        voice_style_dir: Directory with voice style JSON files (F1..M5).
        device: torch device.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = "onnx",
        checkpoint: Optional[Union[str, Path]] = None,
        voice_style_dir: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ):
        from .convert import load_models

        model_dir = Path(model_dir)
        with open(model_dir / "tts.json") as f:
            self.cfgs = json.load(f)
        self.text_processor = UnicodeProcessor(model_dir / "unicode_indexer.json")
        self.device = torch.device(device)

        models = load_models(
            checkpoint_path=checkpoint,
            onnx_dir=None if checkpoint else model_dir,
            device=device,
        )
        self.text_encoder = models["text_encoder"]
        self.duration_predictor = models["duration_predictor"]
        self.vector_estimator = models["vector_estimator"]
        self.vocoder = models["vocoder"]

        self.sample_rate = self.cfgs["ae"]["sample_rate"]
        self.base_chunk_size = self.cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = self.cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = self.cfgs["ttl"]["latent_dim"]

        self.voice_style_dir = (
            Path(voice_style_dir)
            if voice_style_dir is not None
            else model_dir.parent / "voice_styles"
        )

    # -- styles ------------------------------------------------------------

    def get_voice_style(self, voice_name: str) -> Style:
        return Style.from_json_file(self.voice_style_dir / f"{voice_name}.json")

    # -- core synthesis ----------------------------------------------------

    def sample_noisy_latent(
        self, duration: np.ndarray, generator: Optional[torch.Generator] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = len(duration)
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = int(
            (duration.max() * self.sample_rate + chunk_size - 1) // chunk_size
        )
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = torch.randn(
            bsz, latent_dim, latent_len, generator=generator, device=self.device
        )
        latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
        latent_mask = torch.from_numpy(
            length_to_mask(latent_lengths, latent_len)
        ).to(self.device)
        return noisy_latent * latent_mask, latent_mask

    @torch.inference_mode()
    def synthesize_chunk(
        self,
        text_list: list[str],
        style: Style,
        total_step: int = DEFAULT_TOTAL_STEPS,
        speed: float = DEFAULT_SPEED,
        lang: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize a batch of short texts. Returns (wav [B, S], dur [B])."""
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"Speed must be in [{MIN_SPEED}, {MAX_SPEED}]")

        bsz = len(text_list)
        text_ids_np, text_mask_np = self.text_processor(text_list, lang)
        text_ids = torch.from_numpy(text_ids_np).to(self.device)
        text_mask = torch.from_numpy(text_mask_np).to(self.device)
        style_ttl = torch.from_numpy(style.ttl).to(self.device)
        style_dp = torch.from_numpy(style.dp).to(self.device)

        duration = self.duration_predictor(text_ids, style_dp, text_mask)
        duration = (duration / speed).cpu().numpy()

        text_emb = self.text_encoder(text_ids, style_ttl, text_mask)

        xt, latent_mask = self.sample_noisy_latent(duration, generator)
        total = torch.full((bsz,), float(total_step), device=self.device)
        for step in range(total_step):
            current = torch.full((bsz,), float(step), device=self.device)
            xt = self.vector_estimator(
                xt, text_emb, style_ttl, text_mask, latent_mask, current, total
            )
        wav = self.vocoder(xt).cpu().numpy()
        return wav, duration

    def synthesize(
        self,
        text: str,
        voice_style: Style,
        total_steps: int = DEFAULT_TOTAL_STEPS,
        speed: float = DEFAULT_SPEED,
        max_chunk_length: Optional[int] = None,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        lang: Optional[str] = UNKNOWN_LANGUAGE,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full text -> waveform synthesis with chunking.

        Returns (wav [1, S], total duration [1]).
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        is_valid, unsupported = self.text_processor.validate_text(text)
        if not is_valid:
            raise ValueError(f"Unsupported character(s): {unsupported}")

        if max_chunk_length is None:
            max_chunk_length = (
                DEFAULT_MAX_CHUNK_LENGTH_KO if lang == "ko" else DEFAULT_MAX_CHUNK_LENGTH
            )
        text_chunks = chunk_text(text, max_chunk_length)

        wav_list, dur_list = [], []
        for text_chunk in text_chunks:
            wav, dur = self.synthesize_chunk(
                [text_chunk], voice_style, total_steps, speed, lang, generator
            )
            wav_list.append(wav)
            dur_list.append(dur)

        silence = np.zeros(
            (1, int(silence_duration * self.sample_rate)), dtype=np.float32
        )
        pieces = []
        for i, wav in enumerate(wav_list):
            pieces.append(wav)
            if i < len(wav_list) - 1:
                pieces.append(silence)
        wav_cat = np.concatenate(pieces, axis=1)
        dur_cat = sum(dur_list) + silence_duration * (len(wav_list) - 1)
        return wav_cat, dur_cat

    __call__ = synthesize

    def save_audio(self, wav: np.ndarray, output_path: Union[str, Path]) -> None:
        import soundfile as sf

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), np.asarray(wav).squeeze(), self.sample_rate)
