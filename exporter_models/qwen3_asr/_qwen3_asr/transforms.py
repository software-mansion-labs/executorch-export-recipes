"""XNNPACK-only source transforms that keep normalisation and RoPE delegated.

XNNPACK has no layer-norm operator, its mean only reduces the spatial dims of
a 4-D tensor, and it does not slice a dynamic-shape tensor. So LayerNorm,
RMSNorm and RoPE's rotate-half would each run on the host. Each one is
rewritten here with the same math expressed through ops XNNPACK takes: the
mean over the last dim as a linear against a constant 1/D row, and
rotate-half as a linear against a constant signed permutation. Vulkan and MLX
fuse the original forms into their own kernels, so they keep them.
"""

import torch
import torch.nn.functional as F
from torch import nn


class LinearMeanLayerNorm(nn.Module):
    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        d = ln.normalized_shape[-1]
        self.eps = ln.eps
        self.weight = nn.Parameter(ln.weight.detach().clone())
        self.bias = nn.Parameter(ln.bias.detach().clone())
        self.register_buffer("avg", torch.full((1, d), 1.0 / d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - F.linear(x, self.avg)
        var = F.linear(xc * xc, self.avg)
        return xc * torch.rsqrt(var + self.eps) * self.weight + self.bias


class LinearMeanRMSNorm(nn.Module):
    def __init__(self, norm: nn.Module):
        super().__init__()
        self.eps = norm.eps
        self.weight = nn.Parameter(norm.weight.detach().clone())
        self.register_buffer("avg", torch.full((1, norm.dim), 1.0 / norm.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(F.linear(x * x, self.avg) + self.eps) * self.weight


class LinearRotaryEmbedding(nn.Module):
    """`hf_apply_rotary_emb` with rotate_half(x) = linear(x, R)."""

    def __init__(self, head_dim: int):
        super().__init__()
        half = head_dim // 2
        r = torch.zeros(head_dim, head_dim)
        for i in range(half):
            r[i, i + half] = -1.0  # out[i] = -x[i + half]
            r[i + half, i] = 1.0  # out[i + half] = x[i]
        self.register_buffer("rotate", r)

    def forward(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_out = q * cos + F.linear(q, self.rotate) * sin
        k_out = k * cos + F.linear(k, self.rotate) * sin
        return q_out, k_out


def _replace(module: nn.Module, match, build) -> int:
    count = 0
    for name, child in module.named_children():
        if match(child):
            setattr(module, name, build(child))
            count += 1
        else:
            count += _replace(child, match, build)
    return count


def xnnpack_audio_encoder(encoder: nn.Module) -> nn.Module:
    _replace(encoder, lambda m: isinstance(m, nn.LayerNorm), LinearMeanLayerNorm)
    return encoder


def xnnpack_text_model(model: nn.Module, head_dim: int) -> nn.Module:
    from executorch.examples.models.llama.norm import RMSNorm

    n = _replace(model, lambda m: type(m) is RMSNorm, LinearMeanRMSNorm)
    assert n > 0, "no RMSNorm found"
    model.rope.apply_rotary_emb = LinearRotaryEmbedding(head_dim)
    return model
