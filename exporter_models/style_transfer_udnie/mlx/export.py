import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_STYLE = "udnie"


def _load_net():
    name = "style_udnie_mlx_net"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_transformer_net.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
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


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    model = _load_net().load_style(_STYLE)
    inputs = (torch.randn(shape, dtype=torch.float32),)
    if variant["precision"] == "fp16":
        model = _Fp16IO(model.half().eval()).eval()
    ep = torch_export(model, inputs)
    program = to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
