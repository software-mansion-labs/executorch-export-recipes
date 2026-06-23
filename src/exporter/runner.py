import importlib.util
from collections.abc import Callable
from pathlib import Path

from .errors import ExporterError, SpecError


def load_export_callable(export_py_path) -> Callable[[dict, str], str]:
    path = Path(export_py_path)
    if not path.exists():
        raise SpecError(f"export script not found: {path}")
    spec = importlib.util.spec_from_file_location(f"exporter_model_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "export", None)
    if not callable(fn):
        raise SpecError(f"export script {path} has no callable 'export(config, out_dir)'")
    return fn


def run_export(export_py_path, config: dict, out_dir: str) -> str:
    fn = load_export_callable(export_py_path)
    try:
        return fn(config, out_dir)
    except ExporterError:
        raise
    except Exception as e:
        raise ExporterError("export_failed", f"{Path(export_py_path)}: {e}") from e
