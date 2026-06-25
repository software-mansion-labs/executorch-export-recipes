import json
import os
from pathlib import Path

from .errors import SpecError
from .spec import Spec, load_spec, load_spec_file


def _models_dir() -> Path:
    env = os.environ.get("EXPORTER_MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "exporter_models"


MODELS_DIR = _models_dir()


def _config_path(model: str) -> Path:
    return MODELS_DIR / model / "config.json"


def pointer_path(model: str) -> Path:
    p = MODELS_DIR / model / "model.json"
    if not p.exists():
        raise SpecError(f"no model.json pointer for model '{model}' at {p}")
    return p


def list_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    out = []
    for d in MODELS_DIR.iterdir():
        if not d.is_dir() or not (d / "export.py").exists():
            continue
        if (d / "config.json").exists() or (d / "model.json").exists():
            out.append(d.name)
    return sorted(out)


def spec_path(model: str) -> Path:
    p = _config_path(model)
    if not p.exists():
        raise SpecError(f"no spec for model '{model}' at {p}")
    return p


def export_script_path(model: str) -> Path:
    p = MODELS_DIR / model / "export.py"
    if not p.exists():
        raise SpecError(f"no export.py for model '{model}' at {p}")
    return p


def _fetch_config(repo: str, config_path: str) -> dict:
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(repo, config_path)
    except Exception as e:
        raise SpecError(f"could not fetch config '{config_path}' from HF repo '{repo}': {e}") from e
    return json.loads(Path(local).read_text())


def _pointer_data(model: str) -> dict:
    return json.loads(pointer_path(model).read_text())


def model_sizes(model: str) -> list[str]:
    p = MODELS_DIR / model / "model.json"
    if not p.exists():
        return []
    return list(json.loads(p.read_text()).get("sizes", []))


def is_sized(model: str) -> bool:
    return bool(model_sizes(model))


def _load_via_pointer(model: str, size: str | None) -> Spec:
    data = _pointer_data(model)
    if "repo" not in data or "config" not in data:
        raise SpecError(f"model.json for '{model}' must have 'repo' and 'config' keys")
    sizes = data.get("sizes")
    config_path = data["config"]
    has_placeholder = "{size}" in config_path

    if sizes:
        if not has_placeholder:
            raise SpecError(f"model '{model}' declares 'sizes' but 'config' has no '{{size}}' placeholder")
        chosen = size or sizes[0]
        if chosen not in sizes:
            raise SpecError(f"size '{chosen}' not in {sizes} for model '{model}'")
        config_path = config_path.replace("{size}", chosen)
    else:
        if has_placeholder:
            raise SpecError(f"model '{model}' config has '{{size}}' placeholder but no 'sizes' list")
        if size is not None:
            raise SpecError(f"model '{model}' has no sizes; --size not applicable")

    return load_spec(_fetch_config(data["repo"], config_path))


def load(model: str, *, local_spec: str | None = None, size: str | None = None) -> Spec:
    if local_spec:
        return load_spec_file(local_spec)
    if _config_path(model).exists():
        if size is not None:
            raise SpecError(f"model '{model}' has a local config and is not sized; --size not applicable")
        return load_spec_file(_config_path(model))
    if (MODELS_DIR / model / "model.json").exists():
        return _load_via_pointer(model, size)
    raise SpecError(f"model '{model}' has neither config.json nor model.json under {MODELS_DIR / model}")
