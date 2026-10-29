import hashlib
import importlib.util
import sys
import urllib.request
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent
_WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy"
_WEIGHTS_SHA256 = "e7567cde013fb64813973ce6e1ecc25a80c05c3ca7adbc5a54f3c3d90991b854"


def load_model():
    key = "u2netp_vendored"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / "_u2net.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    weights = _MODEL_DIR / "weights" / "u2netp.pth"
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_WEIGHTS_URL, weights)
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if actual != _WEIGHTS_SHA256:
        raise ValueError(f"U2NETP checkpoint sha256 {actual} != {_WEIGHTS_SHA256}; delete {weights} and retry")
    model = sys.modules[key].U2NETP(3, 1)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval()


class U2NetPForRNE(torch.nn.Module):
    """Accept RGB in [0,1]; return the fused foreground logit map."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        # Upstream ToTensorLab(flag=0) divides by the image's maximum, not 255.
        # The clamp defines the previously undefined all-black input as zeros.
        image = image / image.amax().clamp_min(1e-6)
        return self.model((image - self.mean) / self.std)
