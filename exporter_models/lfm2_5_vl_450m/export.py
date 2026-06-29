import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = "LiquidAI/LFM2.5-VL-450M"


def _load_pkg():
    pkg_name = f"lfm2_5_vl_{_HERE.name}"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    pkg_init = _HERE / "_lfm2_5_vl" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        pkg_init,
        submodule_search_locations=[str(_HERE / "_lfm2_5_vl")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    _load_pkg()
    pkg_name = f"lfm2_5_vl_{_HERE.name}"
    export_mod = importlib.import_module(f"{pkg_name}.export_lfm2_5_vl")
    export_all = export_mod.export_all
    params_path = export_mod._resolve_params_path(_MODEL_DIR, None)

    out_path = str(Path(out_dir) / config["variant"]["file"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        export_all(
            model_dir=_MODEL_DIR,
            output=out_path,
            quantize=True,
            params_path=params_path,
        )
    finally:
        os.chdir(cwd)
    return out_path
