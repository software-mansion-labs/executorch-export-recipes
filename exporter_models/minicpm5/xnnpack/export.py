import importlib.util
from pathlib import Path


def export(config: dict, out_dir: str) -> str:
    spec = importlib.util.spec_from_file_location(
        "minicpm5_export", Path(__file__).resolve().parent.parent / "_export.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export(config, out_dir)
