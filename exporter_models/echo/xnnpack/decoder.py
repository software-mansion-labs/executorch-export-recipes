"""Exports the Higgs codec decoder to ExecuTorch XNNPACK, react-native-executorch's contract:
forward(codes i64[8, T]) -> audio f32[T * 960], T 4-<max frames>, with the get_model_schema method.

The graph is laid out for XNNPACK:
- Every convolution is a 2-D one on [1, C, 1, T]. A 1-D convolution is lowered as unsqueeze, convolution, squeeze, and
  each of those pairs costs a transpose into channels-last and one back; 4-D tensors stay channels-last throughout.
- The stages from fp32_from on (the tail in the shipped program) compute in fp32, the ones before in fp16 (XNNPACK's fp16 accumulates in fp16: the
  last stages write the waveform, and their rounding lands in the audio).
- In the fp32 stages Snake, x + sin(a x)^2 / a, runs as z + sin(z)^2 on z = a x: the convolution before it is scaled
  by a where it feeds only the Snake, the one after it by 1 / a. In fp16 that would push the channels with a small a
  into fp16's subnormals, so those stages keep x + sign(a) (sin(a x) / sqrt|a|)^2.

The weights stay fp32 in the file, twice the size they compute at: XNNPACK takes fp16 weights only for a convolution
whose input and output are fp32 (validate_datatypes_with_bias), and refuses the program otherwise.
"""
import importlib.util
import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

ECHO = Path(__file__).resolve().parent.parent / "_echo"
# The decoder in order: each block is its Snake and upsampling convolution, then its three residual units.
STAGES = ["conv1"] + [f"block{i}{part}" for i in range(5) for part in ("", ".unit1", ".unit2", ".unit3")] + ["tail", "none"]
HOP = 960


def module(name: str):
    spec = importlib.util.spec_from_file_location(f"echo_{name}", ECHO / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


codec_mod, hub = module("codec"), module("hub")


def channels(t: torch.Tensor) -> torch.Tensor:
    """A per-channel vector as [1, C, 1, 1]."""
    return t.detach().reshape(1, -1, 1, 1).clone()


class Conv(nn.Module):
    """A Conv1d as a 2-D convolution over [1, C, 1, T], its inputs scaled per channel by `before` and its outputs by
    `after` (folded into the weights)."""

    def __init__(self, conv: nn.Conv1d, before: Optional[torch.Tensor] = None, after: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        weight, bias = conv.weight.detach().clone(), conv.bias.detach().clone()
        if before is not None:
            weight = weight * before.reshape(1, -1, 1)
        if after is not None:
            weight, bias = weight * after.reshape(-1, 1, 1), bias * after.reshape(-1)
        self.weight, self.bias = nn.Parameter(weight.unsqueeze(2), requires_grad=False), nn.Parameter(bias, requires_grad=False)
        self.padding, self.dilation = (0, conv.padding[0]), (1, conv.dilation[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, padding=self.padding, dilation=self.dilation)


class Upsample(nn.Module):
    """A ConvTranspose1d as a 2-D transposed convolution, its inputs scaled per channel by `before`."""

    def __init__(self, conv: nn.ConvTranspose1d, before: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        weight = conv.weight.detach().clone()  # [in, out, taps]
        if before is not None:
            weight = weight * before.reshape(-1, 1, 1)
        self.weight = nn.Parameter(weight.unsqueeze(2), requires_grad=False)
        self.bias = nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
        self.stride, self.padding, self.extra = (1, conv.stride[0]), (0, conv.padding[0]), (0, conv.output_padding[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, output_padding=self.extra)


class Snake(nn.Module):
    """x + sin(a x)^2 / a for fp16: as x + sign(a) (sin(a x) / sqrt|a|)^2, since 1 / a overflows fp16 for the smallest a."""

    def __init__(self, alpha: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("alpha", channels(alpha))
        self.register_buffer("rsqrt", channels((alpha.abs() + 1e-9).rsqrt()))
        self.register_buffer("sign", channels(alpha.sign()))
        self.register_buffer("inv", channels((alpha + 1e-9).reciprocal()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float32:  # experiment: the plain form
            return x + self.inv * torch.sin(self.alpha * x).pow(2)
        r = torch.sin(self.alpha * x) * self.rsqrt
        return x + self.sign * (r * r)


def wave(z: torch.Tensor) -> torch.Tensor:
    """Snake on z = a x, up to the factor 1 / a that the next convolution carries."""
    s = torch.sin(z)
    return z + s * s


def inverse(alpha: torch.Tensor) -> torch.Tensor:
    """1 / a, without Snake's 1e-9: it scales x itself here, where the 1e-9 would cost up to 0.25% (the smallest a is
    4e-7). The sin^2 term changes by 1e-9 / a in exchange, below 1e-5 from a = 1e-4 up."""
    return alpha.detach().reshape(-1).reciprocal()


class Unit(nn.Module):
    """A residual unit: x + conv2(snake2(conv1(snake1(x))))."""

    def __init__(self, unit: nn.Module, folded: bool) -> None:
        super().__init__()
        self.folded = folded
        a1, a2 = unit.snake1.alpha.detach().reshape(-1), unit.snake2.alpha.detach().reshape(-1)
        if folded:
            self.register_buffer("scale", channels(a1))
            self.conv1, self.conv2 = Conv(unit.conv1, inverse(a1), a2), Conv(unit.conv2, inverse(a2))
        else:
            self.snake1, self.snake2 = Snake(a1), Snake(a2)
            self.conv1, self.conv2 = Conv(unit.conv1), Conv(unit.conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.folded:
            return x + self.conv2(wave(self.conv1(wave(x * self.scale))))
        return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))


class Up(nn.Module):
    """A block's Snake and its upsampling transposed convolution."""

    def __init__(self, block: nn.Module, folded: bool) -> None:
        super().__init__()
        self.folded = folded
        alpha = block.snake1.alpha.detach().reshape(-1)
        if folded:
            self.register_buffer("scale", channels(alpha))
            self.up = Upsample(block.conv_t1, inverse(alpha))
        else:
            self.snake, self.up = Snake(alpha), Upsample(block.conv_t1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(wave(x * self.scale) if self.folded else self.snake(x))


class Tail(nn.Module):
    """The final Snake and the convolution to one channel. The codec's last layer, named tanh, is an identity."""

    def __init__(self, decoder: nn.Module, folded: bool) -> None:
        super().__init__()
        assert isinstance(decoder.tanh, nn.Identity)
        self.folded = folded
        alpha = decoder.snake1.alpha.detach().reshape(-1)
        if folded:
            self.register_buffer("scale", channels(alpha))
            self.conv = Conv(decoder.conv2, inverse(alpha))
        else:
            self.snake, self.conv = Snake(alpha), Conv(decoder.conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(wave(x * self.scale) if self.folded else self.snake(x))


class Decoder(nn.Module):
    """codec.decode() for one clip: codes [8, T] -> audio [T * 960]. Stages before `fp32_from` compute in fp16."""

    def __init__(self, codec: nn.Module, fp32_from: str = "tail", fold: bool = True) -> None:
        super().__init__()
        first = STAGES.index(fp32_from)
        acoustic = codec.acoustic_decoder
        self.quantizers, self.fc2 = codec.quantizer.quantizers, codec.fc2
        stages: List[nn.Module] = [Conv(acoustic.conv1)]
        for block in acoustic.block:
            stages.append(Up(block, folded=fold and len(stages) >= first))
            for unit in (block.res_unit1, block.res_unit2, block.res_unit3):
                stages.append(Unit(unit, folded=fold and len(stages) >= first))
        stages.append(Tail(acoustic, folded=fold and len(stages) >= first))
        assert len(stages) == len(STAGES) - 1
        self.stages = nn.ModuleList(stages)
        self.first = first
        # The lookups and their projections follow the first stage's precision.
        if first > 0:
            for part in [self.quantizers, self.fc2] + list(self.stages[:first]):
                part.half()

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        q = self.quantizers
        x = q[0].project_out(q[0].codebook.decode(codes[0]))  # [T, 1024]
        for i in range(1, len(q)):
            x = x + q[i].project_out(q[i].codebook.decode(codes[i]))
        # [T, 256] -> [1, 256, 1, T]: the layout XNNPACK's convolutions read as [1, 1, T, 256], with nothing to move.
        x = self.fc2(x).view(1, 1, -1, 256).permute(0, 3, 1, 2)
        for i, stage in enumerate(self.stages):
            if i == self.first:
                x = x.float()
            x = stage(x)
        return x.float().reshape(-1)


def schema(max_frames: int) -> str:
    frames = {"kind": "range", "range": {"min": 4, "max": max_frames, "step": 1}}
    samples = {"kind": "range", "range": {"min": 4 * HOP, "max": max_frames * HOP, "step": HOP}}
    return json.dumps({"forward": {
        "inputs": [{"kind": "Tensor", "dtype": "int64", "shape": [{"kind": "constant", "value": 8}, frames]}],
        "outputs": [{"kind": "Tensor", "dtype": "float32", "shape": [samples]}],
        "runtimeConstraints": [{
            "kind": "linear", "coefficients": [HOP, 0],
            "dimLhs": {"paramSide": "output", "tensorIdx": 0, "dimIdx": 0},
            "dimRhs": {"paramSide": "input", "tensorIdx": 0, "dimIdx": 1},
        }],
    }})


def fresh_codec() -> nn.Module:
    return codec_mod.drop_training_buffers(codec_mod.load_codec(hub.codec()))


def lower(ep, max_frames: int) -> bytes:
    import collections

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    edge = to_edge_transform_and_lower(
        {"forward": ep}, partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        constant_methods={"get_model_schema": schema(max_frames)},
    )
    kinds = collections.Counter(
        "delegate" if "executorch_call_delegate" in str(node.target) else getattr(node.target, "__name__", str(node.target))
        for node in edge.exported_program("forward").graph.nodes
        if node.op == "call_function" and "getitem" not in str(node.target)
    )
    print("  " + ", ".join(f"{count} {kind}" for kind, count in kinds.most_common()), flush=True)
    return edge.to_executorch().buffer


def build(max_frames: int = 512) -> bytes:
    """The shipped program: the tail in fp32, up to 512 frames (20.48 s; XNNPACK's memory grows with it)."""
    with torch.no_grad():
        decoder = Decoder(fresh_codec()).eval()
        frames = torch.export.Dim("T", min=4, max=max_frames)
        ep = torch.export.export(decoder, (torch.randint(0, 1024, (8, 120)),), dynamic_shapes=({1: frames},), strict=False)
        return lower(ep, max_frames)
