import importlib.util
import re
import sys
from collections.abc import Callable
from pathlib import Path

from .errors import ExporterError, SpecError


def load_export_callable(export_py_path) -> Callable[[dict, str], str | None]:
    path = Path(export_py_path)
    if not path.exists():
        raise SpecError(f"export script not found: {path}")
    # Unique, importable module name: torch.export/dynamo re-imports the module
    # that defines any traced nn.Module by name, so it must live in sys.modules.
    # Derived from the full script dir so nested/umbrella models can't collide.
    mod_name = "exporter_model_" + re.sub(r"[^0-9a-zA-Z]+", "_", str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, "export", None)
    if not callable(fn):
        raise SpecError(f"export script {path} has no callable 'export(config, out_dir)'")
    return fn


# The script's return value is advisory — results are reported from the spec's
# variant files, not from what export() returns.
def run_export(export_py_path, config: dict, out_dir: str) -> str | None:
    fn = load_export_callable(export_py_path)
    try:
        return fn(config, out_dir)
    except ExporterError:
        raise
    except Exception as e:
        raise ExporterError("export_failed", f"{Path(export_py_path)}: {e}") from e
