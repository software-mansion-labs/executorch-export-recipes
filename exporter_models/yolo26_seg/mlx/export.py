import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_wrapper():
    pkg_init = _MODEL_DIR / "_wrappers" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "yolo26_seg_wrappers_mlx", pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_wrappers")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["yolo26_seg_wrappers_mlx"] = mod
    spec.loader.exec_module(mod)
    return mod



class _Fp16IO(torch.nn.Module):
    """fp32 I/O, fp16 compute — mirrors CoreML FLOAT16 semantics so the app's
    fp32 input/output contract is preserved while the graph runs in half."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args):
        args = tuple(
            a.to(torch.float16) if torch.is_floating_point(a) else a for a in args
        )
        out = self.inner(*args)
        if isinstance(out, (tuple, list)):
            return tuple(
                o.to(torch.float32) if torch.is_floating_point(o) else o for o in out
            )
        return out.to(torch.float32)


# Precision follows the variant. fp16 uses the _Fp16IO wrapper so the graph runs
# in half while the app's fp32 input/output contract is preserved, matching the
# CoreML FLOAT16 semantics. The MPSGraph fp32 abort documented in
# exporter_models/yolo26/coreml/export.py is a Core ML compiler problem and does
# not apply to MLX, which lowers this graph at either precision.
def export(config: dict, out_dir: str) -> str:
    from executorch import exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from ultralytics import YOLO

    YOLOInstanceSegmentation = _load_wrapper().YOLOInstanceSegmentation

    size = config.get("size") or "n"
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    resolution = shape[-1]

    yolo = YOLO(f"yolo26{size}-seg.pt")
    model = YOLOInstanceSegmentation(yolo.model, method_size=resolution).eval()
    inputs = (torch.randn(*shape, dtype=torch.float32),)
    with torch.no_grad():
        model(*inputs)  # warmup: fix export-mode head shapes

    if variant["precision"] == "fp16":
        model = _Fp16IO(model.half().eval()).eval()

    ep = torch_export(model, inputs)
    program = exir.to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
