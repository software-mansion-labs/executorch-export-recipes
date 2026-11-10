"""Exportable LFM3-VL vision tower + projector for one fixed-size image.

Re-implements `Lfm3VlVisionModel` + `Lfm3VlMultiModalProjector`
(LiquidAI/LFM3-VL-2B, `modeling_lfm3_vl.py`) for a static [1, 3, H, W] input,
as RNE's multimodal runner expects (`vision_encoder: f32[1,3,H,W] ->
f32[1, visualTokens, hidden]`). Differences from upstream, all exact up to
float rounding:

- Preprocessing is in the graph: RNE feeds raw 0..255 RGB (alpha=1, beta=0),
  and the upstream processor does x/255, then (x - 0.5) / 0.5.
- Images are one frame, which upstream duplicates along the temporal patch
  axis before the patchifier. The two temporal slices of the patchifier
  weight are summed instead, and the patchifier becomes a stride-16 conv.
- Tokens stay in raster order through the tower and are reordered into the
  upstream merge-block order only before the projector. Attention is
  bidirectional and RoPE uses each token's own (h, w), so the order does not
  change any per-token result.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class VisionConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    layer_types: list
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    conv_kernel_size: int
    rope_parameters: dict
    norm_eps: float = 1e-5
    in_channels: int = 3

    @classmethod
    def from_hf(cls, hf: dict) -> "VisionConfig":
        v = hf["vision_config"]
        return cls(**{k: v[k] for k in cls.__dataclass_fields__ if k in v})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = (xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.weight.type_as(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        d, hd = cfg.hidden_size, self.head_dim
        self.q_proj = nn.Linear(d, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.out_proj = nn.Linear(self.n_heads * hd, d, bias=False)
        self.q_layernorm = RMSNorm(hd, cfg.norm_eps)
        self.k_layernorm = RMSNorm(hd, cfg.norm_eps)

    def forward(self, x, cos, sin):
        N = x.shape[0]
        hd = self.head_dim
        q = self.q_layernorm(self.q_proj(x).view(N, self.n_heads, hd))
        k = self.k_layernorm(self.k_proj(x).view(N, self.n_kv, hd))
        v = self.v_proj(x).view(N, self.n_kv, hd)
        c, s = cos[:, None, :], sin[:, None, :]
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        rep = self.n_heads // self.n_kv
        q = q.transpose(0, 1)                                   # [H, N, hd]
        k = k.transpose(0, 1).repeat_interleave(rep, dim=0)
        v = v.transpose(0, 1).repeat_interleave(rep, dim=0)
        att = torch.softmax((q @ k.transpose(-1, -2)) * self.scaling, dim=-1)
        y = (att @ v).transpose(0, 1).reshape(N, self.n_heads * hd)
        return self.out_proj(y)


class VisionConv(nn.Module):
    """Gated depthwise 3x3 conv over the raster patch grid."""

    def __init__(self, cfg: VisionConfig, grid_h: int, grid_w: int):
        super().__init__()
        d = cfg.hidden_size
        self.grid = (grid_h, grid_w)
        self.B_proj = nn.Linear(d, d, bias=False)
        self.C_proj = nn.Linear(d, d, bias=False)
        self.x_proj = nn.Linear(d, d, bias=False)
        k = cfg.conv_kernel_size
        self.spatial_conv = nn.Conv2d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        gh, gw = self.grid
        gated = self.B_proj(x) * self.x_proj(x)                     # [N, d]
        grid = gated.transpose(0, 1).reshape(1, -1, gh, gw)
        conv = self.spatial_conv(grid).reshape(-1, gh * gw).transpose(0, 1)
        return self.out_proj(self.C_proj(x) * conv)


class VisionLayer(nn.Module):
    def __init__(self, cfg: VisionConfig, idx: int, grid_h: int, grid_w: int):
        super().__init__()
        self.is_attention = cfg.layer_types[idx] == "full_attention"
        if self.is_attention:
            self.self_attn = VisionAttention(cfg)
        else:
            self.conv = VisionConv(cfg, grid_h, grid_w)
        self.feed_forward = MLP(cfg.hidden_size, cfg.intermediate_size)
        self.operator_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)

    def forward(self, x, cos, sin):
        h = self.operator_norm(x)
        h = self.self_attn(h, cos, sin) if self.is_attention else self.conv(h)
        x = x + h
        return x + self.feed_forward(self.ffn_norm(x))


class Projector(nn.Module):
    def __init__(self, cfg: VisionConfig, hf: dict):
        super().__init__()
        f = hf["downsample_factor"]
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=hf["projector_layer_norm_eps"])
        self.linear_1 = nn.Linear(cfg.hidden_size * f * f, hf["projector_hidden_size"], bias=hf["projector_bias"])
        self.linear_2 = nn.Linear(hf["projector_hidden_size"], hf["text_config"]["hidden_size"], bias=hf["projector_bias"])

    def forward(self, x):
        x = self.layer_norm(x).reshape(-1, self.linear_1.in_features)
        return self.linear_2(F.gelu(self.linear_1(x)))


class VisionEncoder(nn.Module):
    """[1, 3, H, W] raw RGB (0..255) -> [1, (H/32)*(W/32), text hidden]."""

    def __init__(self, hf: dict, image_hw=(512, 512)):
        super().__init__()
        cfg = VisionConfig.from_hf(hf)
        self.cfg = cfg
        p, m = cfg.patch_size, cfg.spatial_merge_size
        H, W = image_hw
        assert H % (p * m) == 0 and W % (p * m) == 0, image_hw
        self.image_hw = image_hw
        self.grid = (H // p, W // p)
        gh, gw = self.grid
        self.patchifier = nn.Conv2d(cfg.in_channels, cfg.hidden_size, kernel_size=p, stride=p, bias=False)
        self.embedding_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.layers = nn.ModuleList(VisionLayer(cfg, i, gh, gw) for i in range(cfg.num_hidden_layers))
        self.projector = Projector(cfg, hf)

        # Axial RoPE (upstream `Lfm3VlVisionRotaryEmbedding`), raster order.
        theta = cfg.rope_parameters["rope_theta"]
        spatial = cfg.hidden_size // cfg.num_attention_heads // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, spatial, 2, dtype=torch.float) / spatial))
        hpos, wpos = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        fh = hpos.flatten().float()[:, None] * inv_freq
        fw = wpos.flatten().float()[:, None] * inv_freq
        freqs = torch.cat([fh, fw], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

        # Raster index -> upstream merge-block order (block-major over m x m).
        idx = torch.arange(gh * gw).view(gh // m, m, gw // m, m).permute(0, 2, 1, 3).flatten()
        self.register_buffer("block_order", idx, persistent=False)

    def forward(self, image):
        x = image / 127.5 - 1.0                                      # rescale + normalize(0.5, 0.5)
        x = self.patchifier(x)                                        # [1, d, gh, gw]
        x = x.flatten(2)[0].transpose(0, 1)                           # [N, d] raster
        x = self.embedding_norm(x)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = x.index_select(0, self.block_order)
        return self.projector(x).unsqueeze(0)


def convert(hf_state: dict, cfg: VisionConfig) -> dict:
    """HF `model.vision_tower.*` / `model.multi_modal_projector.*` -> VisionEncoder."""
    d = cfg.hidden_size
    out = {}
    for name, t in hf_state.items():
        if name.startswith("model.multi_modal_projector."):
            out["projector." + name[len("model.multi_modal_projector."):]] = t
            continue
        if not name.startswith("model.vision_tower."):
            continue
        key = name[len("model.vision_tower."):]
        if key == "patchifier.weight":
            # [d, C * T * p * p] -> sum the T duplicated slices -> conv [d, C, p, p]
            p, T, C = cfg.patch_size, cfg.temporal_patch_size, cfg.in_channels
            out[key] = t.float().view(d, C, T, p, p).sum(2)
        elif key.endswith("conv.in_proj.weight"):
            base = key[: -len("in_proj.weight")]
            b, c, x = t.split(d, dim=0)
            out[base + "B_proj.weight"], out[base + "C_proj.weight"], out[base + "x_proj.weight"] = b, c, x
        else:
            out[key] = t
    return out
