"""Face-mesh weight fetcher.

The checkpoint is a TFLite -> PyTorch convert of Google's `face_landmark.tflite`
done by https://github.com/tiqq111/mediapipe_pytorch (see its `tflite2pt.py`).
There is no release artifact, so it is pulled straight from the repo at a pinned
commit and cached under `~/.cache/exporter/facemesh/`.
"""

import hashlib
import urllib.request
from pathlib import Path

_CACHE = Path.home() / ".cache" / "exporter" / "facemesh"

#: mediapipe_pytorch @ main, 2021-03-15 (the repo has had no commit since).
COMMIT = "4eb026c8fdcdceda28cb7d96bec64c150577f289"
URL = (
    f"https://raw.githubusercontent.com/tiqq111/mediapipe_pytorch/{COMMIT}"
    "/facial_landmarks/model_weights/facial_landmarks.pth"
)
SHA256 = "43183e63193445731be64855c13f8db2a4182193503f319ae6fe741292f1fc91"


def checkpoint_path() -> Path:
    """Download the checkpoint if needed, verify its digest, return its path."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    target = _CACHE / "facial_landmarks.pth"
    if not target.exists():
        tmp = target.with_suffix(".pth.part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{target}: sha256 {digest}, expected {SHA256}")
    return target
