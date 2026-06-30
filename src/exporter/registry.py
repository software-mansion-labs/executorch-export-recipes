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
        if not d.is_dir():
            continue
        has_top_export = (d / "export.py").exists()
        # backend-split models keep export.py inside each backend subdir
        has_backend_export = any(
            (d / b / "export.py").exists() for b in model_backends(d.name)
        )
        if not (has_top_export or has_backend_export):
            continue
        if (d / "config.json").exists() or (d / "model.json").exists():
            out.append(d.name)
    return sorted(out)


def export_script_path(model: str, backend: str | None = None) -> Path:
    backends = model_backends(model)
    if backends:
        chosen = backend or backends[0]
        p = MODELS_DIR / model / chosen / "export.py"
    else:
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


def is_multi_pte(model: str) -> bool:
    p = MODELS_DIR / model / "model.json"
    if not p.exists():
        return False
    return bool(json.loads(p.read_text()).get("multi_pte", False))


def model_backends(model: str) -> list[str]:
    p = MODELS_DIR / model / "model.json"
    if not p.exists():
        return []
    return list(json.loads(p.read_text()).get("backends", []))


def is_backend_split(model: str) -> bool:
    return bool(model_backends(model))


def _load_via_pointer(model: str, size: str | None, backend: str | None) -> Spec:
    data = _pointer_data(model)
    if "config" not in data:
        raise SpecError(f"model.json for '{model}' must have a 'config' key")
    config_path = data["config"]

    sizes = data.get("sizes")
    if sizes:
        if "{size}" not in config_path and "{size}" not in data.get("repo", ""):
            raise SpecError(f"model '{model}' declares 'sizes' but neither 'config' nor 'repo' has a '{{size}}' placeholder")
        chosen = size or sizes[0]
        if chosen not in sizes:
            raise SpecError(f"size '{chosen}' not in {sizes} for model '{model}'")
        config_path = config_path.replace("{size}", chosen)
    elif "{size}" in config_path:
        raise SpecError(f"model '{model}' config has '{{size}}' placeholder but no 'sizes' list")
    elif size is not None:
        raise SpecError(f"model '{model}' has no sizes; --size not applicable")

    backends = data.get("backends")
    if backends:
        if "{backend}" not in config_path:
            raise SpecError(f"model '{model}' declares 'backends' but 'config' has no '{{backend}}' placeholder")
        chosen_b = backend or backends[0]
        if chosen_b not in backends:
            raise SpecError(f"backend '{chosen_b}' not in {backends} for model '{model}'")
        local = MODELS_DIR / model / chosen_b / "config.json"
        if local.exists():
            return load_spec_file(local)
        config_path = config_path.replace("{backend}", chosen_b)
    elif "{backend}" in config_path:
        raise SpecError(f"model '{model}' config has '{{backend}}' placeholder but no 'backends' list")
    elif backend is not None:
        raise SpecError(f"model '{model}' has no backends; --backend not applicable")

    if "repo" not in data:
        raise SpecError(f"model.json for '{model}' must have 'repo' and 'config' keys")
    repo = data["repo"]
    if sizes:
        repo = repo.replace("{size}", chosen)
    return load_spec(_fetch_config(repo, config_path))


def load(model: str, *, local_spec: str | None = None, size: str | None = None,
         backend: str | None = None) -> Spec:
    if local_spec:
        return load_spec_file(local_spec)
    if _config_path(model).exists():
        if size is not None:
            raise SpecError(f"model '{model}' has a local config and is not sized; --size not applicable")
        if backend is not None:
            raise SpecError(f"model '{model}' has a local config; --backend not applicable")
        return load_spec_file(_config_path(model))
    if (MODELS_DIR / model / "model.json").exists():
        return _load_via_pointer(model, size, backend)
    raise SpecError(f"model '{model}' has neither config.json nor model.json under {MODELS_DIR / model}")
