import importlib.util
import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent.parent


def _load_lib():
    name = "lfm3_text_xnnpack"
    spec = importlib.util.spec_from_file_location(
        name, _MODEL_DIR / "_lfm3" / "__init__.py",
        submodule_search_locations=[str(_MODEL_DIR / "_lfm3")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module(f"{name}.export_lib")


def export(config: dict, out_dir: str) -> str:
    return _load_lib().export(config, out_dir, "xnnpack")
