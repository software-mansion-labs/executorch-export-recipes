import os
from pathlib import Path

from .errors import SpecError
from .spec import Spec, load_spec_file


def _models_dir() -> Path:
    env = os.environ.get("EXPORTER_MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "exporter_models"


MODELS_DIR = _models_dir()


def list_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(
        d.name for d in MODELS_DIR.iterdir()
        if d.is_dir() and (d / "config.json").exists() and (d / "export.py").exists()
    )


def spec_path(model: str) -> Path:
    p = MODELS_DIR / model / "config.json"
    if not p.exists():
        raise SpecError(f"no spec for model '{model}' at {p}")
    return p


def export_script_path(model: str) -> Path:
    p = MODELS_DIR / model / "export.py"
    if not p.exists():
        raise SpecError(f"no export.py for model '{model}' at {p}")
    return p


def load(model: str, *, local_spec: str | None = None) -> Spec:
    if local_spec:
        return load_spec_file(local_spec)
    return load_spec_file(spec_path(model))
