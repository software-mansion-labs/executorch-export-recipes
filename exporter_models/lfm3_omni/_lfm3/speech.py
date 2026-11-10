# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exportable Omni speech heads and offline LiquidCodec decoder.

Architecture adapted from LiquidAI/LFM3-Omni-2B at
21d75123f86c1f7ffb4bef1ea202710c4e7ebe67 (HuggingFace, Apache-2.0).
The inverse FFT is an exact real-valued linear transform followed by overlap-add;
no complex-valued tensors or runtime FFT operators are required.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .model import FeedForward, RMSNorm, rotate_half


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg['num_attention_heads']
        self.kv_heads = cfg['num_key_value_heads']
        self.dim = cfg['head_dim']
        h, d = cfg['hidden_size'], self.dim
        self.q_proj = nn.Linear(h, self.heads * d, bias=False)
        self.k_proj = nn.Linear(h, self.kv_heads * d, bias=False)
        self.v_proj = nn.Linear(h, self.kv_heads * d, bias=False)
        self.o_proj = nn.Linear(self.heads * d, h, bias=False)
        self.q_norm = RMSNorm(d, cfg['norm_eps'])
        self.k_norm = RMSNorm(d, cfg['norm_eps'])
        self.gate_proj = nn.Linear(h, self.heads, bias=False)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, t, self.heads, self.dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).reshape(b, t, self.kv_heads, self.dim)).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.kv_heads, self.dim).transpose(1, 2)
        q = (q.float() * cos + rotate_half(q.float()) * sin).to(x.dtype)
        k = (k.float() * cos + rotate_half(k.float()) * sin).to(x.dtype)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.dim ** -0.5 + mask
        out = torch.matmul(scores.float().softmax(-1).to(x.dtype), v).transpose(1, 2)
        out = out * self.gate_proj(x).sigmoid().unsqueeze(-1)
        return self.o_proj(out.reshape(b, t, -1))


class ShortConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg['hidden_size']
        self.width = cfg['conv_L_cache']
        self.in_proj = nn.Linear(h, 3 * h, bias=False)
        self.conv = nn.Conv1d(h, h, self.width, groups=h, bias=False)
        self.out_proj = nn.Linear(h, h, bias=False)

    def forward(self, x):
        b, c, v = self.in_proj(x).chunk(3, -1)
        conv = self.conv(F.pad((b * v).transpose(1, 2), (self.width - 1, 0)))
        return self.out_proj(c * conv.transpose(1, 2))


class Layer(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.kind = kind
        self.sandwich = cfg['use_sandwich_norm']
        h, eps = cfg['hidden_size'], cfg['norm_eps']
        self.operator_norm = RMSNorm(h, eps)
        self.ffn_norm = RMSNorm(h, eps)
        self.feed_forward = FeedForward(h, cfg['intermediate_size'])
        if kind == 'conv':
            self.conv = ShortConv(cfg)
        else:
            self.self_attn = Attention(cfg)
        if self.sandwich:
            self.post_operator_norm = RMSNorm(h, eps)
            self.post_ffn_norm = RMSNorm(h, eps)

    def forward(self, x, cos, sin, mask):
        h = self.operator_norm(x)
        h = self.conv(h) if self.kind == 'conv' else self.self_attn(h, cos, sin, mask)
        x = x + (self.post_operator_norm(h) if self.sandwich else h)
        h = self.feed_forward(self.ffn_norm(x))
        return x + (self.post_ffn_norm(h) if self.sandwich else h)


class Trunk(nn.Module):
    def __init__(self, cfg, *, bidirectional=False):
        super().__init__()
        self.cfg = cfg
        self.bidirectional = bidirectional
        self.layers = nn.ModuleList(Layer(cfg, kind) for kind in cfg['layer_types'])
        self.embedding_norm = RMSNorm(cfg['hidden_size'], cfg['norm_eps'])
        p = next(iter(cfg['rope_parameters'].values()))
        self.theta = p['rope_theta']

    def forward(self, x):
        t = x.shape[1]
        dim = self.cfg['head_dim']
        freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(self.theta) / dim))
        angles = torch.arange(t, dtype=torch.float32)[:, None] * freq[None]
        angles = torch.cat([angles, angles], -1)[None, None]
        # Upstream rounds the rotary tables to compute dtype before fp32 rotation.
        cos, sin = angles.cos().to(x.dtype).float(), angles.sin().to(x.dtype).float()
        i = torch.arange(t)[:, None]
        j = torch.arange(t)[None]
        visible = torch.ones((t, t), dtype=torch.bool) if self.bidirectional else (j <= i)
        if not self.bidirectional and 'sliding_attention' in self.cfg['layer_types']:
            visible = visible & (i - j < self.cfg['sliding_window'])
        mask = torch.where(visible, 0.0, float('-inf')).to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.embedding_norm(x)


class HeadNorm(RMSNorm):
    def forward(self, x):
        h = x.float()
        return (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


class Codebook(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg['codebook_size'], cfg['depth_config']['hidden_size'])
        self.embedding_norm = HeadNorm(cfg['depth_config']['hidden_size'], cfg['depth_config']['norm_eps'])


class DepthDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n, d = cfg['num_codebooks'], cfg['depth_config']['hidden_size']
        self.n, self.d = n, d
        self.depth_linear = nn.Linear(cfg['text_config']['hidden_size'], n * d)
        self.mask_embedding = nn.Parameter(torch.zeros(d))
        self.model = Trunk(cfg['depth_config'], bidirectional=True)
        self.depth_embeddings = nn.ModuleList(Codebook(cfg) for _ in range(n))

    def forward(self, hidden, codes, masked):
        dtype = self.depth_linear.weight.dtype
        x = self.depth_linear(hidden.to(dtype)).reshape(1, self.n, self.d)
        embeds = torch.stack([e.embedding(codes[:, i]) for i, e in enumerate(self.depth_embeddings)], 1)
        x = self.model(x + torch.where(masked[..., None], self.mask_embedding, embeds))
        return torch.stack([F.linear(e.embedding_norm(x[:, i]), e.embedding.weight) for i, e in enumerate(self.depth_embeddings)], 1).float()


class AudioEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.codebook_embedding = nn.Embedding(cfg['num_codebooks'] * cfg['codebook_size'], cfg['text_config']['hidden_size'])
        self.register_buffer('offsets', torch.arange(cfg['num_codebooks']) * cfg['codebook_size'], persistent=False)

    def forward(self, codes):
        return self.codebook_embedding(codes + self.offsets).sum(-2).unsqueeze(0).float()


class Codec(nn.Module):
    """Decode 1..256 codec frames offline; each becomes 1920 PCM samples."""
    def __init__(self):
        super().__init__()
        cfg = dict(hidden_size=512, intermediate_size=2304, num_hidden_layers=8,
                   num_attention_heads=8, num_key_value_heads=2, head_dim=64,
                   norm_eps=1e-5, conv_L_cache=7, use_sandwich_norm=True,
                   layer_types=['conv','conv','sliding_attention','conv','sliding_attention','conv','sliding_attention','conv'],
                   sliding_window=30, rope_parameters={'sliding_attention': {'rope_theta': 1000000.0}})
        self.overlap_add = False
        self.project_out = nn.Linear(48, 256)
        self.from_quantizer = nn.Linear(256, 512)
        self.trunk = Trunk(cfg)
        self.lin = nn.Linear(512, 1282)
        self.register_buffer('basis', torch.tensor([1, 8, 56, 336]), persistent=False)
        self.register_buffer('levels', torch.tensor([8, 7, 6, 6]), persistent=False)
        n = torch.arange(1280, dtype=torch.float64)
        k = torch.arange(641, dtype=torch.float64)
        angle = 2 * math.pi * k[:, None] * n[None] / 1280
        mult = torch.full((641, 1), 2.0, dtype=torch.float64); mult[0] = mult[-1] = 1
        window = torch.hann_window(1280, dtype=torch.float64)
        self.register_buffer('inverse_dft', (torch.cat([mult * angle.cos(), -mult * angle.sin()]) * window / 1280).float().unsqueeze(1))
        self.register_buffer('envelope_window', window.square().float().reshape(1, 1, -1))

    def forward(self, codes):
        # Layout [1,12,F] -> [1,F,12,4]. EOS frames must be removed by caller.
        c = codes.transpose(1, 2).unsqueeze(-1)
        half = self.levels // 2
        if self.overlap_add:
            # EXPORT PATCH: this MLX lowering treats integer floor division as
            # true division. Codes are nonnegative and <=2015, so explicit
            # fp32 floor operations decode the exact mixed-radix FSQ digits.
            half = torch.floor(self.levels.float() / 2)
            quotient = torch.floor(c.float() / self.basis.float())
            digits = quotient - torch.floor(quotient / self.levels.float()) * self.levels.float()
            latents = ((digits - half) / half).reshape(1, codes.shape[-1], 48)
        else:
            latents = (((c // self.basis) % self.levels - half) / half).reshape(1, codes.shape[-1], 48)
        dtype = self.project_out.weight.dtype
        x = self.from_quantizer(self.project_out(latents.to(dtype)))
        x = x.unsqueeze(2).expand(-1, -1, 6, -1).reshape(1, codes.shape[-1] * 6, 512)
        x = self.lin(self.trunk(x)).transpose(1, 2).float()
        mag, phase = x.chunk(2, 1)
        mag = mag.exp()
        spec = torch.cat([mag * phase.cos(), mag * phase.sin()], 1)
        if self.overlap_add:
            # EXPORT PATCH: MLX transposed convolution materializes a >18 GB
            # temporary even for one codec frame. The exact inverse DFT is a
            # per-window matrix multiply followed by four overlapping hops.
            windows = spec.transpose(1, 2) @ self.inverse_dft[:, 0, :]
            audio = self._overlap_add(windows)
            envelope = self._overlap_add(self.envelope_window.expand_as(windows))
        else:
            audio = F.conv_transpose1d(spec, self.inverse_dft, stride=320)
            envelope = F.conv_transpose1d(torch.ones_like(spec[:, :1]), self.envelope_window, stride=320)
        return (audio[:, 0, 480:-480] / envelope[:, 0, 480:-480]).float()

    @staticmethod
    def _overlap_add(windows):
        hops = windows.reshape(1, -1, 4, 320)
        return sum(F.pad(hops[:, :, i, :].reshape(1, 1, -1),
                         (i * 320, (3 - i) * 320)) for i in range(4))
