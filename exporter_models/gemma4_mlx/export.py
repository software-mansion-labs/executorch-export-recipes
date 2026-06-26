import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_standalone():
    p = _HERE / "_export_mlx_standalone.py"
    spec = importlib.util.spec_from_file_location("gemma4_mlx_standalone", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemma4_mlx_standalone"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    standalone = _load_standalone()
    out_path = str(Path(out_dir) / config["variant"]["file"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = standalone.resolve_checkpoint(None)
    standalone.export(model_key="gemma4-e2b", out_path=out_path, checkpoint=checkpoint)
    return out_path
