"""EasyOCR fused (CRAFT detect + CRNN recognize) bucketed export — Vulkan.
The model code is vendored under `easyocr/_easyocr`; this thin entrypoint resolves
the variant + language and lowers to out_dir. Top-level imports stay stdlib+torch
only so the harness can discover `export` without a backend env (torch/executorch/
torchao load inside the vendored builder)."""
import sys
from pathlib import Path

import torch  # noqa: F401  (declares the torch dependency for the harness)

_MODEL_DIR = Path(__file__).resolve().parent.parent
_BACKEND = "vulkan"


def export(config: dict, out_dir: str) -> str:
    if str(_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(_MODEL_DIR))
    import _easyocr.builder as builder

    lang = config["size"]
    variant = config.get("variant") or config["variants"][0]
    out = (Path(out_dir) / variant["file"]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    return builder.build_and_lower(_BACKEND, lang, str(out))
