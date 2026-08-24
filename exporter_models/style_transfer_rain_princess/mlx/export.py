# MLX: exports, but the fp16 artifact produces NON-FINITE output.
#
# History worth recording, because the first diagnosis was wrong. This exporter
# used to fail at export with:
#
#     Operator torch._ops.aten.relu_.default is not in Core ATen opset
#
# That was NOT a model limitation -- this exporter was simply missing the compile
# config every working MLX exporter in this repo uses:
#
#     EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)
#
# Added 2026-08-20. The export now succeeds. But executing the resulting fp16
# .pte under the fork runtime yields non-finite values (NaN/Inf) in the output
# image tensor [1, 3, 640, 640], so it is still NOT shippable and "mlx" stays out
# of model.json backends.
#
# Contrast with the sibling models fixed by the same one-line change, which export
# AND execute correctly and ARE published: efficientnet_v2_s, ssdlite320_mobilenet_
# v3_large, rfdetr_nano_detector, rfdetr_nano_segmentation. So the compile config
# was a real blocker across the board; style transfer has a second, separate
# problem on top of it.
#
# Likely lead, untested: this network is fp32-scale-sensitive in a way the others
# are not -- it is InstanceNorm after every conv, and the int8 work on this same
# model already showed its activation scales are easy to wreck (see the calibration
# note in xnnpack/export.py). An fp32 MLX variant is the obvious next experiment;
# the config currently declares fp16 only.

import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_STYLE = "rain_princess"


def _load_net():
    name = "style_rain_princess_mlx_net"
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
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

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
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
