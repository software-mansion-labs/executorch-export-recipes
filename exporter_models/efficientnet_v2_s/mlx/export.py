from pathlib import Path

import torch
import torchvision.models as tvm
from torch.export import export as torch_export

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class _EfficientNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # The shipped .pte normalizes the raw [0,1] image inside the graph.
        return self.model((x - self.mean) / self.std)


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

    model = _EfficientNetWrapper(
        tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.DEFAULT)
    ).eval()
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
