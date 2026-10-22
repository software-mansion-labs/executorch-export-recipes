import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_STYLE = "rain_princess"


def _load_net():
    name = "style_rain_princess_coreml_net"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_transformer_net.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]

    model = _load_net().load_style(_STYLE)
    inputs = (torch.randn(shape, dtype=torch.float32),)
    ep = torch_export(model, inputs)
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
        # Pin the compute unit: this architecture is pathologically slow on the
        # ANE. Measured on an iPhone 16 (ET 1.4.1, 3 warmup + 15 timed, median,
        # mean over the four styles):
        #
        #   fp16 ALL           209.4 ms   <- ANE-eligible, and the slowest
        #   fp16 CPU_ONLY      109.7 ms
        #   fp16 CPU_AND_GPU   114.2 ms   <- shipped
        #   fp32 ALL           152.8 ms
        #
        # ExecuTorch's default is ComputeUnit.ALL, under which fp16 is
        # ANE-eligible and fp32 is not (the ANE is fp16-only). That made fp32
        # look faster than fp16 and is why its removal was briefly contested:
        # the comparison was measuring placement, not precision. Excluding the
        # ANE makes fp16 the fastest build at half of fp32's size.
        #
        # CPU_AND_GPU over CPU_ONLY despite CPU_ONLY's slightly better mean:
        # CPU_ONLY ranges 99.6-119.9 ms across the four styles while
        # CPU_AND_GPU holds ~114 ms on all four, so it should degrade less
        # under CPU contention and thermal load.
        compute_unit=ct.ComputeUnit.CPU_AND_GPU,
    )
    program = to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
