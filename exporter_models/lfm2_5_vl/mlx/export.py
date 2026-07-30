import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR_PATH = _HERE.parent

_MODEL_DIRS = {
    "450m": "LiquidAI/LFM2.5-VL-450M",
    "1_6b": "LiquidAI/LFM2.5-VL-1.6B",
}


def _load_pkg():
    pkg_name = f"lfm2_5_vl_{_MODEL_DIR_PATH.name}_mlx"
    if pkg_name in sys.modules:
        return pkg_name
    pkg_init = _MODEL_DIR_PATH / "_lfm2_5_vl" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        pkg_init,
        submodule_search_locations=[str(_MODEL_DIR_PATH / "_lfm2_5_vl")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return pkg_name


def export(config: dict, out_dir: str) -> str:
    model_dir = _MODEL_DIRS[config.get("size") or "450m"]

    pkg_name = _load_pkg()
    export_mod = importlib.import_module(f"{pkg_name}.export_lfm2_5_vl_mlx")
    export_all = export_mod.export_all
    params_path = export_mod._resolve_params_path(model_dir, None)

    out_path = str(Path(out_dir) / config["variant"]["file"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        export_all(
            model_dir=model_dir,
            output=out_path,
            quantize=True,
            params_path=params_path,
        )
    finally:
        os.chdir(cwd)
    return out_path
