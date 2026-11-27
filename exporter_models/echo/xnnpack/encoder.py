"""Exports the Higgs codec encoder to ExecuTorch XNNPACK, react-native-executorch's contract:
forward(wav f32[frames, 960]) -> codes i64[8, frames] at 24 kHz, frames 25-<max frames>, with the get_model_schema
method of the shipped Core ML and Vulkan encoders.

The graph is laid out for XNNPACK on a phone:
- Every convolution is a 2-D one on [1, C, 1, T], so nothing leaves channels-last (a 1-D convolution costs a transpose
  in and one out).
- The DAC encoder computes in fp32. Snake, x + sin(a x)^2 / a, runs as z + sin(z)^2 on z = a x, the convolutions
  around it scaled by a and 1 / a. Convolutions with more than 4 MB of weights are split by output channels (the last
  block otherwise runs at 60% of the others' rate: its weights are streamed from memory once per few positions).
- HuBERT's first convolution and its GroupNorm compute in fp32 (the squares overflow fp16), like every LayerNorm; its
  other convolutions, the attention and the semantic encoder compute in fp16.
- The transformer's linears run on the int8 QAT checkpoint's integers, their inputs quantized per frame (XNNPACK's
  dynamically quantized matmuls, 2.3x the fp16 rate); the residual stream between them stays fp32.
- The 24 -> 16 kHz resampler is a strided convolution with a constant filter; the mean of the 13 hidden states and the
  every-second-frame cut are folded into the semantic encoder's first convolution (stride 2, dilation 2).
- The residual quantizer picks argmax(x e^T - |e|^2 / 2) per codebook; argmax and the codebook lookup run outside
  XNNPACK (8 small delegates around them).
- Memory: the operators keep the traced order (ExecuTorch otherwise interleaves the two branches), and the output of
  HuBERT's first convolution is held below the DAC's first block in XNNPACK's workspace (see Encoder.embed).

fp16 on a phone is not fp16 on an M-series Mac: there XNNPACK's fp16 kernels accumulate more precisely, and the DAC's
blocks 1-4 in fp16 (dac_fp16 = 1,2,3,4) look fine (99.7% of codebook 0) where a Snapdragon 8 Gen 3 gives 97.7%. Score
fp16 changes on the phone, not on a Mac.
"""
import collections
import importlib.util
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import torchao  # noqa: F401 - registers torchao's quantization ops

HERE = Path(__file__).resolve().parent
ECHO = HERE.parent / "_echo"
HOP = 960
MIN_FRAMES = 25


def module(path: Path):
    spec = importlib.util.spec_from_file_location(f"echo_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


codec_mod, hub = module(ECHO / "codec.py"), module(ECHO / "hub.py")


def frozen(t: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(t.detach().clone().contiguous(), requires_grad=False)


def channels(t: torch.Tensor) -> nn.Parameter:
    """A per-channel vector as [1, C, 1, 1]."""
    return frozen(t.reshape(1, -1, 1, 1))


def rows(x: torch.Tensor) -> torch.Tensor:
    """[1, C, 1, T] -> [T, C]."""
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def planes(x: torch.Tensor) -> torch.Tensor:
    """[T, C] -> [1, C, 1, T]."""
    return x.view(1, 1, -1, x.shape[1]).permute(0, 3, 1, 2)


# A convolution's weights are streamed once per few output positions: past the caches' size the fp32 DAC's last block
# ran at 135 GFLOPS on a Snapdragon 8 Gen 3 against 230 for the others. Larger weights are split by output channels.
CHUNK_BYTES = 4 << 20


def joined(parts: List[torch.Tensor]) -> torch.Tensor:
    """Concatenates along channels, at most four tensors at a time (ExecuTorch's XNNPACK concat takes up to five)."""
    while len(parts) > 1:
        parts = [torch.cat(parts[i:i + 4], dim=1) if len(parts[i:i + 4]) > 1 else parts[i] for i in range(0, len(parts), 4)]
    return parts[0]


class Conv(nn.Module):
    """A Conv1d as a 2-D convolution over [1, C, 1, T], its inputs scaled per channel by `before` and its outputs by
    `after` (folded into the weights). `chunked`: weights above CHUNK_BYTES run as several convolutions, each writing a
    slice of the output channels."""

    def __init__(
        self,
        conv: nn.Conv1d,
        before: Optional[torch.Tensor] = None,
        after: Optional[torch.Tensor] = None,
        padding: Optional[int] = None,
        stride: Optional[int] = None,
        dilation: Optional[int] = None,
        chunked: bool = False,
    ) -> None:
        super().__init__()
        weight = conv.weight.detach().clone()
        bias = conv.bias.detach().clone() if conv.bias is not None else None
        if before is not None:
            assert conv.groups == 1
            weight = weight * before.reshape(1, -1, 1)
        if after is not None:
            weight = weight * after.reshape(-1, 1, 1)
            bias = bias * after.reshape(-1) if bias is not None else None
        weight = weight.unsqueeze(2)
        count = 1
        if chunked and conv.groups == 1:
            while weight.numel() * 4 / count > CHUNK_BYTES and count < weight.shape[0]:
                count *= 2
        self.weights = nn.ParameterList(frozen(part) for part in weight.chunk(count))
        self.biases = nn.ParameterList(frozen(part) for part in bias.chunk(count)) if bias is not None else None
        self.stride = (1, stride or conv.stride[0])
        self.padding = (0, conv.padding[0] if padding is None else padding)
        self.dilation = (1, dilation or conv.dilation[0])
        self.groups = conv.groups

    @property
    def weight(self) -> torch.Tensor:
        return self.weights[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return joined([
            F.conv2d(x, weight, self.biases[i] if self.biases is not None else None, self.stride, self.padding, self.dilation, self.groups)
            for i, weight in enumerate(self.weights)
        ])


def quantize_rows(x: torch.Tensor) -> torch.Tensor:
    """x through int8, one asymmetric scale per row: XNNPACK then runs the linears that read it as int8 matmuls."""
    block = [1] * (x.dim() - 1) + [x.shape[-1]]
    scale, zero = torch.ops.torchao.choose_qparams_affine(x, "ASYMMETRIC", block, torch.int8, -128, 127, None, None, torch.int8)
    x = torch.ops.torchao.quantize_affine(x, block, scale, zero, torch.int8, -128, 127)
    return torch.ops.torchao.dequantize_affine(x, block, scale, zero, torch.int8, -128, 127)


class Linear(nn.Module):
    """A Linear, its outputs scaled by `gain`. With `quantized`, on the QAT checkpoint's int8 weights (one scale per
    output), reading its input through int8."""

    def __init__(self, linear: nn.Linear, quantized: bool = False, gain: float = 1.0) -> None:
        super().__init__()
        self.quantized = quantized
        self.bias = frozen(linear.bias * gain)
        if quantized:
            values, scale, bits = linear.qat
            assert bits == 8 and scale.shape[1] == 1, "int8 with one scale per output"
            self.values = frozen(values)
            self.scale = frozen(scale.reshape(-1, 1).float() * gain)
            self.zero = frozen(torch.zeros_like(self.scale, dtype=torch.int8))
        else:
            self.weight = frozen(linear.weight * gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            return F.linear(x, self.weight, self.bias)
        block = [1, self.values.shape[1]]
        weight = torch.ops.torchao.dequantize_affine(self.values, block, self.scale, self.zero, torch.int8, -128, 127)
        return F.linear(quantize_rows(x), weight, self.bias)


class Mean(nn.Module):
    """The mean of [n, d] rows as a matmul with 1 / d, plus eps: [n, 1]. XNNPACK's own mean wants a 4-D tensor."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = frozen(torch.full((1, width), 1.0 / width))
        self.bias = frozen(torch.tensor([eps]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class Norm(nn.Module):
    """LayerNorm over the rows of [n, d], always in fp32: HuBERT's features reach 700, and their squares overflow fp16."""

    def __init__(self, norm: nn.LayerNorm) -> None:
        super().__init__()
        width = norm.normalized_shape[0]
        self.mean, self.var = Mean(width, 0.0), Mean(width, norm.eps)
        self.weight, self.bias = frozen(norm.weight), frozen(norm.bias)

    def half(self) -> "Norm":
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide = x.float()
        d = wide - self.mean(wide)
        return (d * torch.rsqrt(self.var(d * d)) * self.weight + self.bias).to(x.dtype)


class Resample(nn.Module):
    """torchaudio's default sinc resampler (the recipe's port) as a strided convolution with the filter as a constant:
    [1, 1, 1, N] -> [1, 1, 1, N new / orig]. Its output channels are the phases, interleaved by the reshape."""

    def __init__(self, orig_freq: int, new_freq: int) -> None:
        super().__init__()
        gcd = math.gcd(orig_freq, new_freq)
        self.orig, self.new = orig_freq // gcd, new_freq // gcd
        kernel, self.width = codec_mod._sinc_kernel(self.orig, self.new)  # [new, 1, 2 width + orig]
        self.kernel = frozen(kernel.unsqueeze(2))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(wav, self.kernel, stride=(1, self.orig), padding=(0, self.width))
        return y.permute(0, 2, 3, 1).reshape(1, 1, 1, -1)


class Features(nn.Module):
    """HuBERT's convolutional feature extractor: [1, 1, 1, N] at 16 kHz -> [1, 512, 1, T] at 50 frames/s. The first
    convolution and its GroupNorm (one group per channel: each channel over time) compute in fp32."""

    def __init__(self, hubert: nn.Module, fp16: bool) -> None:
        super().__init__()
        layers = hubert.feature_extractor.conv_layers
        norm = layers[0].layer_norm
        assert norm.num_groups == norm.num_channels and all(not hasattr(layer, "layer_norm") for layer in layers[1:])
        self.first = Conv(layers[0].conv, padding=160)  # the codec pads the 16 kHz audio by 160 on both sides
        self.weight, self.bias, self.eps = channels(norm.weight), channels(norm.bias), norm.eps
        self.rest = nn.ModuleList(Conv(layer.conv) for layer in layers[1:])
        self.fp16 = fp16
        if fp16:
            self.rest.half()

    def statistics(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The GroupNorm as a scale and a shift per channel, [1, 512, 1, 1], from one pass of the first convolution."""
        x = self.first(audio)
        mean = x.mean((2, 3), keepdim=True)
        var = (x * x).mean((2, 3), keepdim=True) - mean * mean
        scale = torch.rsqrt(var + self.eps) * self.weight
        return scale, self.bias - mean * scale

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """`x` is a second pass of the first convolution: read once, it is normalised in place."""
        x = x * scale + shift
        if self.fp16:
            x = x.half()
        x = F.gelu(x)
        for conv in self.rest:
            x = F.gelu(conv(x))
        return x


class Layer(nn.Module):
    """A HuBERT encoder layer (post-norm) over [T, 768]."""

    def __init__(self, layer: nn.Module, quantized: bool, fp16_attention: bool) -> None:
        super().__init__()
        attention = layer.attention
        self.heads, self.fp16_attention = attention.num_heads, fp16_attention
        self.q = Linear(attention.q_proj, quantized, gain=attention.scaling)
        self.k, self.v = Linear(attention.k_proj, quantized), Linear(attention.v_proj, quantized)
        self.out = Linear(attention.out_proj, quantized)
        self.norm1, self.norm2 = Norm(layer.layer_norm), Norm(layer.final_layer_norm)
        self.up = Linear(layer.feed_forward.intermediate_dense, quantized)
        self.down = Linear(layer.feed_forward.output_dense, quantized)

    def split(self, x: torch.Tensor) -> torch.Tensor:
        """[T, 768] -> [heads, T, 64]."""
        x = x.view(-1, self.heads, x.shape[1] // self.heads).permute(1, 0, 2)
        return x.half() if self.fp16_attention else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.split(self.q(x)), self.split(self.k(x)), self.split(self.v(x))
        weights = torch.softmax(torch.bmm(q, k.permute(0, 2, 1)), dim=-1)
        a = torch.bmm(weights, v).permute(1, 0, 2).reshape(-1, x.shape[1])
        x = self.norm1(x + self.out(a.to(x.dtype)))
        return self.norm2(x + self.down(F.gelu(self.up(x))))


class Hubert(nn.Module):
    """HuBERT from its first convolution's output (and the GroupNorm's scale and shift) -> the sum of its 13 hidden
    states, [1, 768, 1, T]."""

    def __init__(self, hubert: nn.Module, fp16: bool, quantized: bool, position_fp16: bool = True) -> None:
        super().__init__()
        assert not hubert.config.do_stable_layer_norm
        self.features = Features(hubert, fp16)
        projection = hubert.feature_projection
        self.feature_norm = Norm(projection.layer_norm) if projection.feat_proj_layer_norm else None
        self.project = Linear(projection.projection)
        position = hubert.encoder.pos_conv_embed.conv
        self.position = Conv(position, padding=0)
        # The codec's layer drops the last output of an even kernel: pad one less on the right instead.
        self.position_pad = (position.padding[0], position.padding[0] - 1)
        self.norm = Norm(hubert.encoder.layer_norm)
        # With int8 linears the residual stream stays fp32 (XNNPACK's int8 matmuls read and write fp32).
        self.stream_fp16 = fp16 and not quantized
        self.layers = nn.ModuleList(Layer(layer, quantized, fp16) for layer in hubert.encoder.layers)
        if fp16 and position_fp16:
            self.position.half()
        if self.stream_fp16:
            self.project.half()
            for layer in self.layers:
                for name, part in layer.named_children():
                    if not isinstance(part, Norm):
                        part.half()

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        x = self.features(x, scale, shift)  # [1, 512, 1, T]
        x = rows(x) if self.stream_fp16 else rows(x).float()
        if self.feature_norm is not None:
            x = self.feature_norm(x)
        x = self.project(x)  # [T, 768]
        at = planes(x).to(self.position.weight.dtype)
        position = F.gelu(self.position(F.pad(at, self.position_pad)))
        x = self.norm(x + rows(position).to(x.dtype))
        total = x
        for layer in self.layers:
            x = layer(x)
            total = total + x
        return planes(total)


class Semantic(nn.Module):
    """The semantic encoder on the sum of HuBERT's hidden states at 50 frames/s. Its first convolution takes the mean
    (1 / 13) and every second frame: a k3 convolution of x[::2] is one of x with stride 2 and dilation 2."""

    def __init__(self, semantic: nn.Module, states: int, down: int) -> None:
        super().__init__()
        first = semantic.conv
        assert first.kernel_size[0] == 3 and first.padding[0] == 1 and first.stride[0] == 1
        mean = torch.full((first.in_channels,), 1.0 / states)
        self.first = Conv(first, before=mean, stride=down, dilation=down, padding=down)
        self.blocks = nn.ModuleList()
        for block in semantic.conv_blocks:
            units = nn.ModuleList(nn.ModuleList([Conv(unit.conv1), Conv(unit.conv2)]) for unit in block.res_units)
            self.blocks.append(nn.ModuleList([units, Conv(block.conv)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x.to(self.first.weight.dtype))
        for units, conv in self.blocks:
            for conv1, conv2 in units:
                x = x + conv2(F.elu(conv1(F.elu(x))))
            x = conv(x)
        return x




def wave(z: torch.Tensor) -> torch.Tensor:
    """Snake on z = a x, up to the factor 1 / a that the next convolution carries."""
    s = torch.sin(z)
    return z + s * s


def alpha(snake: nn.Module) -> torch.Tensor:
    return snake.alpha.detach().reshape(-1)


class Snake(nn.Module):
    """x + sin(a x)^2 / a for fp16, as x + sign(a) (sin(a x) / sqrt|a|)^2: 1 / a overflows fp16 for the smallest a."""

    def __init__(self, a: torch.Tensor) -> None:
        super().__init__()
        self.alpha, self.rsqrt, self.sign = channels(a), channels((a.abs() + 1e-9).rsqrt()), channels(a.sign())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sin(self.alpha * x) * self.rsqrt
        return x + self.sign * (r * r)


class Unit(nn.Module):
    """A DAC residual unit: x + conv2(snake2(conv1(snake1(x)))). `folded` (fp32 only): Snake as wave() on z = a x,
    the convolutions scaled by a and 1 / a."""

    def __init__(self, unit: nn.Module, folded: bool) -> None:
        super().__init__()
        a1, a2 = alpha(unit.snake1), alpha(unit.snake2)
        self.folded = folded
        if folded:
            self.scale = channels(a1)
            self.conv1 = Conv(unit.conv1, before=a1.reciprocal(), after=a2, chunked=True)
            self.conv2 = Conv(unit.conv2, before=a2.reciprocal())
        else:
            self.snake1, self.snake2 = Snake(a1), Snake(a2)
            self.conv1, self.conv2 = Conv(unit.conv1), Conv(unit.conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.folded:
            return x + self.conv2(wave(self.conv1(wave(x * self.scale))))
        return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))


class Down(nn.Module):
    """A DAC encoder block: three residual units, Snake, and the strided convolution. The stages named in `fp16`
    ("u1", "u2", "u3", "down") compute in fp16."""

    def __init__(self, block: nn.Module, fp16: Tuple[str, ...] = ()) -> None:
        super().__init__()
        self.units = nn.ModuleList(Unit(getattr(block, f"res_unit{i}"), f"u{i}" not in fp16) for i in (1, 2, 3))
        for i, unit in enumerate(self.units):
            if f"u{i + 1}" in fp16:
                unit.half()
        a = alpha(block.snake1)
        self.folded = "down" not in fp16
        if self.folded:
            self.scale = channels(a)
            self.conv = Conv(block.conv1, before=a.reciprocal(), chunked=True)
        else:
            self.snake, self.conv = Snake(a).half(), Conv(block.conv1).half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for unit in self.units:
            x = unit(x.to(unit.conv1.weight.dtype))
        x = x.to(self.conv.weight.dtype)
        return self.conv(wave(x * self.scale)) if self.folded else self.conv(self.snake(x))


class Acoustic(nn.Module):
    """The DAC encoder: [1, 1, 1, N] at 24 kHz -> [1, 256, 1, N / 960] in fp32. `fp16` names stages to compute in
    fp16, a block ("3") or one of its stages ("0.u3", "0.down"): experiments, every one costs codes on a phone."""

    def __init__(self, encoder: nn.Module, fp16: Tuple[str, ...] = ()) -> None:
        super().__init__()
        self.conv1 = Conv(encoder.conv1)
        stages = ("u1", "u2", "u3", "down")
        self.blocks = nn.ModuleList(
            Down(block, tuple(stage for stage in stages if str(i) in fp16 or f"{i}.{stage}" in fp16))
            for i, block in enumerate(encoder.block)
        )
        a = alpha(encoder.snake1)
        self.scale = channels(a)
        self.conv2 = Conv(encoder.conv2, before=a.reciprocal(), chunked=True)

    def start(self, x: torch.Tensor) -> torch.Tensor:
        """The first convolution and block 0: the stage with the largest tensors, [1, 64, 1, N] in fp32."""
        return self.blocks[0](self.conv1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks[1:]:
            x = block(x)
        return self.conv2(wave(x.float() * self.scale))


class Encoder(nn.Module):
    def __init__(
        self, codec: nn.Module, frames: Tuple[int, ...], hubert_fp16: bool = True, quantized: bool = False,
        dac_fp16: Tuple[str, ...] = (), position_fp16: bool = True, semantic_fp16: bool = True,
    ) -> None:
        super().__init__()
        cfg = codec.config
        acoustic = codec.acoustic_encoder
        # The codec pads the 24 kHz input when the DAC encoder would come out short of the semantic frames.
        assert all(codec._get_conv1d_output_lengths(f * cfg.hop_length, acoustic) == f for f in frames)
        self.resample = Resample(cfg.sample_rate, cfg.semantic_sample_rate)
        self.hubert = Hubert(codec.semantic_model, hubert_fp16, quantized, position_fp16)
        states = len(codec.semantic_model.encoder.layers) + 1
        self.semantic = Semantic(codec.encoder_semantic, states, cfg.semantic_downsample_factor)
        if hubert_fp16 and semantic_fp16:
            self.semantic.half()
        self.acoustic = Acoustic(acoustic, dac_fp16)
        self.fc = Linear(codec.fc)
        quantizers = codec.quantizer.quantizers
        self.project_in = nn.ModuleList(Linear(q.project_in) for q in quantizers)
        self.project_out = nn.ModuleList(Linear(q.project_out) for q in quantizers)
        self.books = len(quantizers)
        for i, q in enumerate(quantizers):
            embed = q.codebook.embed.detach()  # [1024, 64]
            setattr(self, f"embed{i}", frozen(embed))
            setattr(self, f"half_norm{i}", frozen(-0.5 * embed.pow(2).sum(-1)))

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        """[frames, 960] -> the embedding the quantizer reads, [frames, 1024] in fp32."""
        x = wav.reshape(1, 1, 1, -1)
        audio = self.resample(x)
        features = self.hubert.features
        scale, shift = features.statistics(audio)
        # XNNPACK's memory planner never places a tensor below the lowest one alive, so equal-size tensors in a chain
        # creep upward: the DAC's block 0 took 9 slots of 117 MB where 3 are alive at once. HuBERT's first
        # convolution's output is the graph's largest tensor, so it is planned first, at offset 0; computed here and
        # read only after block 0, it is alive below every tensor of block 0, which then packs into the gaps above it.
        first = features.first(audio)
        acoustic = self.acoustic.start(x)
        semantic = self.semantic(self.hubert(first, scale, shift))
        acoustic = self.acoustic(acoustic)
        return self.fc(rows(torch.cat([acoustic, semantic.float()], dim=1)))

    def quantize(self, residual: torch.Tensor) -> torch.Tensor:
        codes: List[torch.Tensor] = []
        for i in range(self.books):
            embed, half_norm = getattr(self, f"embed{i}"), getattr(self, f"half_norm{i}")
            index = F.linear(self.project_in[i](residual), embed, half_norm).argmax(-1)
            residual = residual - self.project_out[i](F.embedding(index, embed))
            codes.append(index)
        return torch.stack(codes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.quantize(self.embed(wav))


def fold_weight_norm(root: nn.Module) -> None:
    from torch.nn.utils import parametrize

    for mod in root.modules():
        if parametrize.is_parametrized(mod):
            for name in list(mod.parametrizations.keys()):
                parametrize.remove_parametrizations(mod, name, leave_parametrized=True)


def load_qat(codec: nn.Module, path: Path) -> None:
    """A QAT checkpoint of encoder_quantization.md into the codec: its floats, and each quantized weight as the float
    its integers and fp16 scales give (one scale per group of columns of the [out, in x taps] layout). Each module
    keeps its integers [out, in x taps], scales [out, groups] and bits as `qat`."""
    from safetensors.torch import load_file

    meta = json.loads((path / "quantization.json").read_text())
    state = load_file(path / "encoder.safetensors")
    weights = {}
    for name, w in meta["weights"].items():
        assert w["symmetric"] and w["granularity"] in ("channel", "group"), (name, w)
        scale, ints = state.pop(f"{name}_scale").float(), state.pop(name)
        if w["bits"] == 4:  # two per byte, the first in the low nibble, each stored as value + 8
            ints = torch.stack([ints & 0xF, ints >> 4], dim=-1).flatten(-2).to(torch.int8) - 8
        ints = ints.reshape(w["shape"][0], -1).to(torch.int8)
        values = ints.view(ints.shape[0], scale.shape[1], -1).float() * scale[..., None]
        weights[name] = (values.reshape(w["shape"]), (ints, scale, w["bits"]))
    unexpected = codec.load_state_dict({k: v.float() for k, v in state.items()}, strict=False).unexpected_keys
    assert not unexpected, unexpected
    for name, (weight, q) in weights.items():
        target = codec.get_submodule(name[: -len(".weight")])
        assert tuple(target.weight.shape) == tuple(weight.shape), (name, target.weight.shape, weight.shape)
        target.weight.data.copy_(weight)
        target.qat = q


def fresh_codec() -> nn.Module:
    """The codec's fp32 weights, the transformer's linears on the int8 QAT checkpoint's integers (their biases are the
    fp32 model's own; QAT learned scales only)."""
    codec = codec_mod.load_codec(hub.codec())
    fold_weight_norm(codec)
    codec.semantic_model.config._attn_implementation = "eager"
    donor = codec_mod.load_codec(hub.codec())
    fold_weight_norm(donor)
    load_qat(donor, Path(hub.encoder_int8()))
    for name, target in codec.semantic_model.encoder.layers.named_modules():
        if isinstance(target, nn.Linear):
            source = donor.semantic_model.encoder.layers.get_submodule(name)
            target.qat = source.qat
            target.weight.data.copy_(source.weight)
    return codec


def schema(max_frames: int) -> str:
    """`get_model_schema` of the shipped dynamic encoders: wav [frames, hop] -> codes [8, frames]."""
    frames = {"kind": "range", "range": {"min": MIN_FRAMES, "max": max_frames, "step": 1}}
    return json.dumps({"forward": {
        "inputs": [{"kind": "Tensor", "dtype": "float32", "shape": [frames, {"kind": "constant", "value": HOP}]}],
        "outputs": [{"kind": "Tensor", "dtype": "int64", "shape": [{"kind": "constant", "value": 8}, frames]}],
        "runtimeConstraints": [{"kind": "equality", "dims": [
            {"paramSide": "input", "tensorIdx": 0, "dimIdx": 0}, {"paramSide": "output", "tensorIdx": 0, "dimIdx": 1}]}],
    }})


def allow_fp16_gelu() -> None:
    """XNNPACK has an fp16 GELU kernel; ExecuTorch's partitioner turns fp16 GELU away."""
    from executorch.backends.xnnpack.partition.config.generic_node_configs import GeluConfig

    GeluConfig.check_constraints = lambda self, node, ep: self.check_common_constraints(node, ep)


def inferred_views():
    """A pass that writes a reshape's one symbolic dim as -1: XNNPACK infers one dim, ExecuTorch's lowering turns a
    reshape with a symbolic dim away (torch.export resolves -1 to an expression of the input's length)."""
    from executorch.exir.pass_base import PassBase, PassResult

    class InferredViews(PassBase):
        def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
            changed = False
            for node in graph_module.graph.nodes:
                if node.op != "call_function" or getattr(node.target, "__name__", "") != "aten.view_copy.default":
                    continue
                shape = list(node.args[1])
                symbolic = [i for i, size in enumerate(shape) if not isinstance(size, int)]
                if len(symbolic) == 1 and -1 not in shape:
                    shape[symbolic[0]] = -1
                    node.args, changed = (node.args[0], shape), True
            graph_module.graph.eliminate_dead_code()
            return PassResult(graph_module, changed)

    return InferredViews()


def in_traced_order() -> None:
    """After cutting each delegate out, ExecuTorch re-sorts the graph breadth-first (torch's legalize_graph), which
    interleaves the DAC and HuBERT branches: their largest tensors are then alive together (601 MB of workspace).
    A stable sort keeps the traced order: one branch after the other."""
    import executorch.exir.lowered_backend_module as lowered
    from torch.fx.passes.tools_common import stable_topological_sort

    lowered.legalize_graph = stable_topological_sort


def lower(ep, max_frames: int) -> bytes:
    backbone = module(HERE / "backbone.py")

    from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS
    from executorch.backends.xnnpack.partition.config.generic_node_configs import EluConfig
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    backbone.merge_until_stable()
    backbone.transpose_in_matmul()
    allow_fp16_gelu()
    in_traced_order()
    # ELU has a serializer and a runtime node; its partitioner config is only commented out of ExecuTorch's list.
    partitioner = XnnpackPartitioner(configs=ALL_PARTITIONER_CONFIGS + [EluConfig])
    edge = to_edge_transform_and_lower(
        {"forward": ep}, transform_passes=[inferred_views()], partitioner=[partitioner],
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


def build(max_frames: int = 500) -> bytes:
    """The shipped program: HuBERT and the semantic encoder in fp16, the transformer's linears in int8, the DAC in fp32,
    25-500 frames (1-20 s)."""
    with torch.no_grad():
        generator = torch.Generator().manual_seed(0)
        checks = {f: torch.randn(f, HOP, generator=generator) * 0.05 for f in (MIN_FRAMES, 137, max_frames)}
        encoder = Encoder(fresh_codec(), tuple(checks), hubert_fp16=True, quantized=True).eval()
        frames = torch.export.Dim("frames", min=MIN_FRAMES, max=max_frames)
        ep = torch.export.export(encoder, (checks[137],), dynamic_shapes=({0: frames},), strict=False)
        return lower(ep, max_frames)
