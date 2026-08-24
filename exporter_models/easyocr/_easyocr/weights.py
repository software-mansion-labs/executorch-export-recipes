"""EasyOCR weight fetcher — the SAME mechanism as `download_weights.sh`: pull the
CRAFT detector + the per-language CRNN `.pth` from the JaidedAI GitHub releases
(zip-per-model) and cache them under `~/.cache/exporter/easyocr/`.

`num_class` is the CRNN recognizer's output vocab V = len(charset) + 1 (CTC blank
at logit 0). It is fixed per language by the gen2 checkpoint and is what the
`recognize_<W>` method's last output dim must be.
"""
import urllib.request
import zipfile
from pathlib import Path

_CACHE = Path.home() / ".cache" / "exporter" / "easyocr"
_BASE = "https://github.com/JaidedAI/EasyOCR/releases/download"

# (zip_url, member .pth) — language-independent VGG-backbone detector.
CRAFT = (f"{_BASE}/pre-v1.1.6/craft_mlt_25k.zip", "craft_mlt_25k.pth")

# lang -> (zip_url, member .pth, num_class V). Mirrors download_weights.sh +
# export_easyocr.LANGUAGE_CONFIGS (V = charset+1, CTC blank).
CRNN = {
    "english":  (f"{_BASE}/v1.3/english_g2.zip",    "english_g2.pth",   97),
    "latin":    (f"{_BASE}/v1.3/latin_g2.zip",      "latin_g2.pth",    352),
    "japanese": (f"{_BASE}/v1.3/japanese_g2.zip",   "japanese_g2.pth", 2215),
    "zh_sim":   (f"{_BASE}/v1.3/zh_sim_g2.zip",     "zh_sim_g2.pth",   6719),
    "korean":   (f"{_BASE}/v1.3/korean_g2.zip",     "korean_g2.pth",   1009),
    "telugu":   (f"{_BASE}/v1.2/telugu.zip",        "telugu.pth",       166),
    "kannada":  (f"{_BASE}/v1.2/kannada.zip",       "kannada.pth",      168),
    "cyrillic": (f"{_BASE}/v1.6.1/cyrillic_g2.zip", "cyrillic_g2.pth",  208),
}

# easyocr.config gen2 model key per language (for charset emission).
EC_KEY = {
    "english": "english_g2", "latin": "latin_g2", "japanese": "japanese_g2",
    "zh_sim": "zh_sim_g2", "korean": "korean_g2", "telugu": "telugu_g2",
    "kannada": "kannada_g2", "cyrillic": "cyrillic_g2",
}


def fetch(url: str, member: str) -> Path:
    """Download+unzip `url`, return the cached `member` path (idempotent)."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    target = _CACHE / member
    if not target.exists():
        zp = _CACHE / f"{member}.zip"
        urllib.request.urlretrieve(url, zp)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(_CACHE)
        zp.unlink()
    return target


def craft_path() -> Path:
    return fetch(*CRAFT)


def crnn_path(lang: str) -> Path:
    url, member, _ = CRNN[lang]
    return fetch(url, member)


def num_class(lang: str) -> int:
    return CRNN[lang][2]
