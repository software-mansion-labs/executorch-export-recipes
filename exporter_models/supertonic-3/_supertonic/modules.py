"""Shared building blocks for the PyTorch port of Supertonic 3.

All modules replicate the exact computation found in the ONNX graphs
(onnx/*.onnx), including masking order, replicate ("edge") padding and
exact-erf GELU, so that outputs match ONNX Runtime numerically.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class NetConv1d(nn.Module):
    """Conv1d wrapped in a ``net`` attribute with "same" replicate padding.

    Matches ONNX subgraphs that Pad(mode=edge) before an unpadded Conv.
    Weight names: ``<name>.net.weight`` / ``<name>.net.bias``.
    """

    def __init__(self, in_ch, out_ch, ksz, dilation=1, groups=1, bias=True,
                 causal=False):
        super().__init__()
        if causal:
            self.pad = ((ksz - 1) * dilation, 0)
        else:
            self.pad = ((ksz - 1) // 2 * dilation,) * 2
        self.net = nn.Conv1d(
            in_ch, out_ch, ksz, dilation=dilation, groups=groups, bias=bias
        )

    def forward(self, x):
        if max(self.pad) > 0:
            x = F.pad(x, self.pad, mode="replicate")
        return self.net(x)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of a [B, C, T] tensor.

    Weight names: ``<name>.norm.weight`` / ``<name>.norm.bias``.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock(nn.Module):
    """Masked ConvNeXt-1D block with replicate padding and exact GELU.

    ``flat_dwconv=True`` registers the depthwise conv weights directly as
    ``dwconv.weight`` (text encoder / DP / vector field naming), otherwise
    as ``dwconv.net.weight`` (vocoder naming).
    """

    def __init__(self, dim, intermediate_dim, ksz, dilation=1, flat_dwconv=True,
                 causal=False):
        super().__init__()
        self._pad = (ksz - 1) // 2 * dilation
        if flat_dwconv:
            self.dwconv = nn.Conv1d(dim, dim, ksz, dilation=dilation, groups=dim)
        else:
            self.dwconv = NetConv1d(
                dim, dim, ksz, dilation=dilation, groups=dim, causal=causal
            )
        self._flat = flat_dwconv
        self.norm = ChannelLayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, 1)
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, 1)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))
        self._linear_pw = False

    def use_linear_pointwise(self):
        """Swap the k=1 pointwise convs for equivalent Linears on the [B, T, C]
        layout the norm already uses, so a Linear quantizer can reach them."""
        for name in ("pwconv1", "pwconv2"):
            conv = getattr(self, name)
            lin = nn.Linear(conv.in_channels, conv.out_channels, bias=conv.bias is not None)
            lin.weight.data = conv.weight.data[:, :, 0].clone()
            if conv.bias is not None:
                lin.bias.data = conv.bias.data.clone()
            setattr(self, name, lin)
        self._linear_pw = True

    def forward(self, x, mask=None):
        # Caller is expected to have applied the mask to x already.
        residual = x
        if self._flat:
            h = F.pad(x, (self._pad, self._pad), mode="replicate")
            h = self.dwconv(h)
        else:
            h = self.dwconv(x)
        if mask is not None:
            h = h * mask
        if self._linear_pw:
            h = self.norm.norm(h.transpose(1, 2))
            h = self.pwconv2(F.gelu(self.pwconv1(h))).transpose(1, 2)
        else:
            h = self.norm(h)
            h = self.pwconv1(h)
            h = F.gelu(h)
            h = self.pwconv2(h)
        h = residual + self.gamma * h
        if mask is not None:
            h = h * mask
        return h


class ConvNeXtStack(nn.Module):
    """A stack of ConvNeXt blocks registered under ``convnext.{i}``."""

    def __init__(self, dim, intermediate_dim, ksz, dilations, flat_dwconv=True):
        super().__init__()
        self.convnext = nn.ModuleList(
            ConvNeXtBlock(dim, intermediate_dim, ksz, d, flat_dwconv) for d in dilations
        )

    def forward(self, x, mask=None):
        for block in self.convnext:
            if mask is not None:
                x = x * mask
            x = block(x, mask)
        return x


class TextEmbedder(nn.Module):
    """Character embedding lookup -> [B, C, T], masked."""

    def __init__(self, n_chars, dim):
        super().__init__()
        self.char_embedder = nn.Embedding(n_chars, dim)

    def forward(self, text_ids, text_mask):
        x = self.char_embedder(text_ids).transpose(1, 2)
        return x * text_mask


# ---------------------------------------------------------------------------
# VITS-style attention encoder (relative positional multi-head attention)
# ---------------------------------------------------------------------------


class RelPosMultiHeadAttention(nn.Module):
    """Self-attention with relative positional embeddings (VITS style)."""

    def __init__(self, channels, n_heads, window_size=4):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.emb_rel_k = nn.Parameter(
            torch.zeros(1, 2 * window_size + 1, self.k_channels)
        )
        self.emb_rel_v = nn.Parameter(
            torch.zeros(1, 2 * window_size + 1, self.k_channels)
        )

    def _get_relative_embeddings(self, emb, length):
        pad_len = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        if pad_len > 0:
            emb = F.pad(emb, (0, 0, pad_len, pad_len))
        return emb[:, start : start + 2 * length - 1]

    @staticmethod
    def _relative_position_to_absolute_position(x):
        # x: [b, h, l, 2l-1] -> [b, h, l, l].
        # Export-friendly formulation: every reshape target is a product of the
        # *input* dims (l·2l, then l·(2l-1)) so torch.export never has to prove
        # a divisibility guard like (2l²+l-1) % (l+1) == 0 (which it cannot
        # factor). Combined with backed_size_oblivious this exports with
        # dynamic sequence length. Numerically identical to the classic VITS
        # pad/reshape trick.
        b, h, l, _ = x.shape
        x = F.pad(x, (0, 1))              # [b, h, l, 2l]
        x = x.reshape(b, h, 2 * l * l)    # [b, h, 2l²]
        x = x[:, :, : 2 * l * l - l]      # drop last l -> [b, h, l·(2l-1)]
        x = x.reshape(b, h, l, 2 * l - 1)
        return x[:, :, :, l - 1 :]        # last l columns -> [b, h, l, l]

    @staticmethod
    def _absolute_position_to_relative_position(x):
        # x: [b, h, l, l] -> [b, h, l, 2l-1].
        b, h, l, _ = x.shape
        x = F.pad(x, (0, l - 1))
        x = x.reshape(b, h, 2 * l * l - l)
        x = F.pad(x, (l, 0))
        x = x.reshape(b, h, l, 2 * l)
        return x[:, :, :, 1:]

    def forward(self, x, attn_mask):
        b, d, t = x.shape
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        query = q.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        key = k.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        value = v.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)

        query = query / math.sqrt(self.k_channels)
        scores = torch.matmul(query, key.transpose(-2, -1))

        key_rel = self._get_relative_embeddings(self.emb_rel_k, t)
        rel_logits = torch.matmul(query, key_rel.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._relative_position_to_absolute_position(rel_logits)

        scores = scores.masked_fill(attn_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)

        output = torch.matmul(p_attn, value)
        rel_weights = self._absolute_position_to_relative_position(p_attn)
        value_rel = self._get_relative_embeddings(self.emb_rel_v, t)
        output = output + torch.matmul(rel_weights, value_rel.unsqueeze(0))

        output = output.transpose(2, 3).contiguous().view(b, d, t)
        return self.conv_o(output)


class FFN(nn.Module):
    def __init__(self, channels, filter_channels):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, 1)
        self.conv_2 = nn.Conv1d(filter_channels, channels, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class AttnEncoder(nn.Module):
    """Transformer encoder used for text / sentence encoding (VITS layout)."""

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers,
                 window_size=4):
        super().__init__()
        self.attn_layers = nn.ModuleList(
            RelPosMultiHeadAttention(hidden_channels, n_heads, window_size)
            for _ in range(n_layers)
        )
        self.norm_layers_1 = nn.ModuleList(
            ChannelLayerNorm(hidden_channels) for _ in range(n_layers)
        )
        self.ffn_layers = nn.ModuleList(
            FFN(hidden_channels, filter_channels) for _ in range(n_layers)
        )
        self.norm_layers_2 = nn.ModuleList(
            ChannelLayerNorm(hidden_channels) for _ in range(n_layers)
        )

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            y = attn(x, attn_mask)
            x = norm1(x + y)
            y = ffn(x, x_mask)
            x = norm2(x + y)
        return x * x_mask


# ---------------------------------------------------------------------------
# Tanh-key attention (style-token attention, GST style)
# ---------------------------------------------------------------------------


class LinearNet(nn.Module):
    """Linear wrapped in a ``linear`` attribute (matches ONNX weight names)."""

    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


class TanhAttention(nn.Module):
    """Multi-head attention with tanh applied to keys, scores scaled by 1/16.

    Heads are formed by chunking the unit dim and stacking on a new leading
    dim, exactly as in the ONNX graph. ``query_mask`` (0/1, [B, Tq, 1])
    zeroes attention rows after the softmax.
    """

    def __init__(self, q_dim, kv_dim, n_units, out_dim, n_heads, scale=16.0):
        super().__init__()
        self.n_heads = n_heads
        self.scale = scale
        self.W_query = LinearNet(q_dim, n_units)
        self.W_key = LinearNet(kv_dim, n_units)
        self.W_value = LinearNet(kv_dim, n_units)
        self.out_fc = LinearNet(n_units, out_dim)

    def forward(self, q, k, v, query_mask=None):
        Q = torch.stack(self.W_query(q).chunk(self.n_heads, dim=-1), dim=0)
        K = torch.stack(self.W_key(k).chunk(self.n_heads, dim=-1), dim=0)
        V = torch.stack(self.W_value(v).chunk(self.n_heads, dim=-1), dim=0)
        scores = torch.matmul(Q, torch.tanh(K.transpose(-2, -1))) / self.scale
        p = F.softmax(scores, dim=-1)
        if query_mask is not None:
            p = p * query_mask.unsqueeze(0)
        out = torch.matmul(p, V)
        out = torch.cat(list(out), dim=-1)
        return self.out_fc(out)


# ---------------------------------------------------------------------------
# Rotary cross-attention (vector field text conditioning)
# ---------------------------------------------------------------------------


class RotaryCrossAttention(nn.Module):
    """Cross-attention with rotary embeddings on normalized positions.

    Positions of both query (latent frames) and key (text tokens) sequences
    are normalized by their valid (mask) lengths and multiplied by learned
    ``theta`` frequencies; rotation is applied to the two halves of every
    64-dim head. Scores are scaled by 1/16.
    """

    def __init__(self, dim, kv_dim, n_heads, rotary_dim=32, max_len=1000, scale=16.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rotary_dim = rotary_dim
        self.scale = scale
        self.W_query = LinearNet(dim, dim)
        self.W_key = LinearNet(kv_dim, dim)
        self.W_value = LinearNet(kv_dim, dim)
        self.out_fc = LinearNet(dim, dim)
        self.register_buffer(
            "increments", torch.arange(max_len, dtype=torch.int64).view(1, max_len, 1)
        )
        self.theta = nn.Parameter(torch.zeros(1, 1, rotary_dim))

    def _angles(self, length, mask):
        # mask: [B, 1, L] -> lengths [B, 1, 1]
        lengths = mask.sum(dim=(1, 2)).view(-1, 1, 1)
        pos = self.increments[:, :length].to(mask.dtype) / lengths  # [B, L, 1]
        ang = pos * self.theta  # [B, L, rotary_dim]
        return torch.sin(ang), torch.cos(ang)

    @staticmethod
    def _rotate(x, sin, cos):
        # x: [h, B, T, head_dim]; sin/cos: [B, T, rotary_dim]
        d = sin.shape[-1]
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def _split_heads(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).permute(2, 0, 1, 3)

    def forward(self, x_t, kv_t, kv_mask, q_mask):
        # x_t: [B, Lq, dim]; kv_t: [B, Lk, kv_dim]
        # kv_mask / q_mask: [B, 1, Lk] / [B, 1, Lq]
        Q = self._split_heads(self.W_query(x_t))
        K = self._split_heads(self.W_key(kv_t))
        V = self._split_heads(self.W_value(kv_t))

        sin_q, cos_q = self._angles(x_t.shape[1], q_mask)
        sin_k, cos_k = self._angles(kv_t.shape[1], kv_mask)
        Q = self._rotate(Q, sin_q, cos_q)
        K = self._rotate(K, sin_k, cos_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        key_mask = kv_mask.transpose(1, 2).unsqueeze(0).transpose(-2, -1)  # [1,B,1,Lk]
        scores = scores.masked_fill(key_mask == 0, float("-inf"))
        p = F.softmax(scores, dim=-1)
        p = torch.where(
            q_mask.transpose(1, 2).unsqueeze(0) == 0, torch.zeros_like(p), p
        )
        out = torch.matmul(p, V)  # [h, B, Lq, head_dim]
        out = out.permute(1, 2, 0, 3).reshape(x_t.shape[0], x_t.shape[1], -1)
        return self.out_fc(out)


# ---------------------------------------------------------------------------
# Time encoder (sinusoidal + Mish MLP)
# ---------------------------------------------------------------------------


class TimeEncoder(nn.Module):
    """t -> sinusoidal(1000*t) -> Linear -> Mish -> Linear."""

    def __init__(self, time_dim=64, hdim=256, t_scale=1000.0):
        super().__init__()
        half = time_dim // 2
        self.t_scale = t_scale
        self.register_buffer("freqs", torch.zeros(1, half))
        self.mlp = nn.ModuleDict(
            {"0": LinearNet(time_dim, hdim), "2": LinearNet(hdim, time_dim)}
        )

    def forward(self, t):
        # t: [B]
        ang = t.view(-1, 1) * self.t_scale * self.freqs  # [B, half]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        h = self.mlp["0"](emb)
        h = h * torch.tanh(F.softplus(h))  # Mish
        return self.mlp["2"](h)
