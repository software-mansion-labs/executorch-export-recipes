import os
import tempfile
from pathlib import Path

_MODEL_DIR = "LiquidAI/LFM2.5-VL-450M"


def export(config: dict, out_dir: str) -> str:
    from executorch.examples.models.lfm2_5_vl.export_lfm2_5_vl_mlx import export_all

    out_path = str(Path(out_dir) / config["variant"]["file"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        export_all(
            model_dir=_MODEL_DIR,
            output=out_path,
            quantize=True,
            vision_backend="xnnpack",
        )
    finally:
        os.chdir(cwd)
    return out_path
