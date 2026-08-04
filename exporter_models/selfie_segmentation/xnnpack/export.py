import hashlib
import importlib.util
import sys
import urllib.request
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_WEIGHTS_DIR = _MODEL_DIR / "weights"


def _load_selfie_segmentation():
    key = "selfie_segmentation_xnnpack_vendored"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _MODEL_DIR / "_selfie_segmentation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def _fetch_weights(ss, checkpoint: str) -> Path:
    path = _WEIGHTS_DIR / checkpoint
    if not path.exists():
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ss.WEIGHTS_URL.format(checkpoint=checkpoint), path)

    expected = ss.WEIGHTS_SHA256[checkpoint]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"checkpoint '{checkpoint}' sha256 {actual} != expected {expected}; "
            f"delete {path} and re-run, or update WEIGHTS_SHA256 if upstream moved"
        )
    return path


def _checkpoint_for(file_name: str) -> str:
    if "landscape" in file_name:
        return "selfie_segmentation_landscape.pth"
    return "selfie_segmentation.pth"


def export(config: dict, out_dir: str) -> str:
    from torch.export import export as torch_export

    from exporter.backends import lower

    ss = _load_selfie_segmentation()
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    _, _, height, width = shape

    model = ss.SelfieSegmentation(width, height)
    weights = _fetch_weights(ss, _checkpoint_for(variant["file"]))
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    inputs = (torch.rand(*shape, dtype=torch.float32),)
    ep = torch_export(model, inputs, strict=True)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
