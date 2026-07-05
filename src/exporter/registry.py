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


def _is_model_dir(model: str) -> bool:
    d = MODELS_DIR / model
    if not (d / "config.json").exists() and not (d / "model.json").exists():
        return False
    if (d / "export.py").exists():
        return True
    # backend-split models keep export.py inside each backend subdir
    return any((d / b / "export.py").exists() for b in model_backends(model))


def list_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    out = []
    for d in MODELS_DIR.iterdir():
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        if _is_model_dir(d.name):
            out.append(d.name)
            continue
        # umbrella dir (e.g. easyocr/): discover nested models one level down
        for sub in d.iterdir():
            if sub.is_dir() and _is_model_dir(f"{d.name}/{sub.name}"):
                out.append(f"{d.name}/{sub.name}")
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


def model_extras(model: str) -> list[str]:
    p = MODELS_DIR / model / "model.json"
    if not p.exists():
        return []
    return list(json.loads(p.read_text()).get("extras", []))


# pyproject [project.optional-dependencies] extra -> the import names it must
# provide. Keep in sync with pyproject.toml. (pip name != import name for some,
# e.g. keras-hub -> keras_hub.)
_EXTRA_IMPORTS = {
    "coreml": ["coremltools"],
    "quant": ["torchao"],
    "detection": ["rfdetr", "ultralytics", "numpy"],
    "diffusion": ["diffusers"],
    "asr": ["numpy", "tqdm"],
    "vad": ["funasr", "numpy"],
    "tts": ["numpy"],
    "moonshine": ["keras", "keras_hub", "einops"],
}


def check_extras(model: str) -> None:
    """Raise SpecError if any package required by the model's extras is missing,
    with a `pip install` hint. No-op for models that declare no extras."""
    import importlib.util

    extras = model_extras(model)
    if not extras:
        return
    missing = []
    for extra in extras:
        for mod in _EXTRA_IMPORTS.get(extra, []):
            if importlib.util.find_spec(mod) is None and mod not in missing:
                missing.append(mod)
    if missing:
        raise SpecError(
            f"model '{model}' needs extras {extras} but these are missing: "
            f"{missing}. Install with: pip install -e \".[{','.join(extras)}]\""
        )


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
        config_path = config_path.replace("{backend}", chosen_b)
    elif "{backend}" in config_path:
        raise SpecError(f"model '{model}' config has '{{backend}}' placeholder but no 'backends' list")
    elif backend is not None:
        raise SpecError(f"model '{model}' has no backends; --backend not applicable")

    # a vendored config at the fully-substituted path wins over the HF fetch
    local = MODELS_DIR / model / config_path
    if local.exists():
        return load_spec_file(local)

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
