"""PP-DocLayoutV3 layout-detector export (xnnpack). The model code is vendored under
`paddleocr_layout/_layout`; this thin entrypoint resolves the variant and lowers
to out_dir. Top-level imports stay stdlib-only so the harness can discover
`export` without a backend env (torch/executorch/transformers load inside the
vendored builder)."""
import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent.parent


def export(config: dict, out_dir: str) -> str:
    if str(_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(_MODEL_DIR))
    import _layout.builder as builder

    variant = config.get("variant") or config["variants"][0]
    out = (Path(out_dir) / variant["file"]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    return builder.build_and_lower("xnnpack", str(out))
