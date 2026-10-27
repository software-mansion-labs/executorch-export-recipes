"""Exportable LFM3-Audio input path: waveform -> text-space audio embeddings.

Re-implements, for one clip at a time, `Lfm3AudioFeatureExtractor._log_mel`,
`Lfm3AudioEncoder.forward` (downsampler + LFM2 trunk) and `Lfm3AudioProjector`
(LiquidAI/LFM3-Audio-2B). The input is a 16 kHz mono waveform whose length is
already right-padded the way the feature extractor pads it (400 + 160 * k
samples); RNE's runner does that padding.

- The STFT is a stride-160 conv1d against a Hann-windowed DFT basis, so the
  whole front end lowers to ordinary conv/matmul ops (no aten.stft).
- The trunk is causal: short convs (kernel 3) and attention restricted to the
  last 32 tokens, exactly the mask upstream builds.
- Every 8 mel frames become one token (three stride-2 convs).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 128
LOG_FLOOR = -12.0
NORM_OFFSET = 4.0
NORM_DIVISOR = 4.0


@dataclass
class EncoderConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    layer_types: list
    conv_L_cache: int
    rope_theta: float
    norm_eps: float
    sliding_window: int
    conv_channels: int
    text_hidden: int

    @classmethod
    def from_hf(cls, hf: dict) -> "EncoderConfig":
        enc = hf["encoder"]
        lfm = enc["lfm"]
        return cls(
            hidden_size=lfm["hidden_size"],
            intermediate_size=_ff_dim(lfm),
            num_hidden_layers=lfm["num_hidden_layers"],
            num_attention_heads=lfm["num_attention_heads"],
            num_key_value_heads=lfm["num_key_value_heads"],
            layer_types=lfm["layer_types"],
            conv_L_cache=lfm["conv_L_cache"],
            rope_theta=lfm["rope_parameters"]["rope_theta"],
            norm_eps=lfm["norm_eps"],
            sliding_window=enc["sliding_window"],
            conv_channels=enc["dim_conv"],
            text_hidden=hf["lfm"]["hidden_size"],
        )


def mel_filters() -> torch.Tensor:
    """[bins, mels] Slaney filters, the same call the feature extractor makes."""
    from transformers.audio_utils import mel_filter_bank

    return torch.from_numpy(
        mel_filter_bank(
            num_frequency_bins=N_FFT // 2 + 1, num_mel_filters=N_MELS, min_frequency=0.0,
            max_frequency=SAMPLE_RATE / 2, sampling_rate=SAMPLE_RATE, norm="slaney", mel_scale="slaney",
        )
    )


def _ff_dim(lfm: dict) -> int:
    """LFM2's `block_auto_adjust_ff_dim`: 2/3 of intermediate_size, times the
    multiplier, rounded up to `block_multiple_of` (768 -> 1792 here)."""
    ff = lfm["intermediate_size"]
    if lfm.get("block_auto_adjust_ff_dim"):
        ff = int(2 * ff / 3)
        if lfm.get("block_ffn_dim_multiplier") is not None:
            ff = int(lfm["block_ffn_dim_multiplier"] * ff)
        m = lfm["block_multiple_of"]
        ff = m * ((ff + m - 1) // m)
    return ff


class LogMel(nn.Module):
    """Upstream `_log_mel`: periodic Hann window, no centering, |STFT|^2,
    Slaney mel filters, log10 clamped at -12, then (x + 4) / 4."""

    def __init__(self, mel_filters: torch.Tensor):
        super().__init__()
        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)
        window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float64)
        angle = 2 * math.pi * k[:, None] * n[None, :] / N_FFT
        basis = torch.cat([window * torch.cos(angle), -window * torch.sin(angle)], dim=0)
        self.register_buffer("basis", basis.float().unsqueeze(1), persistent=False)
        self.register_buffer("mel_filters", mel_filters.float().T.contiguous(), persistent=False)

    def forward(self, wave):                                         # [1, L]
        spec = F.conv1d(wave.unsqueeze(1), self.basis, stride=HOP)    # [1, 2*bins, frames]
        re, im = spec.chunk(2, dim=1)
        power = re * re + im * im
        mel = torch.matmul(self.mel_filters, power)                   # [1, 128, frames]
        mel = torch.clamp(torch.log10(mel), min=LOG_FLOOR)
        return (mel + NORM_OFFSET) / NORM_DIVISOR


class Downsampler(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        c = cfg.conv_channels
        conv = {"kernel_size": 3, "stride": 2, "padding": 1}
        self.conv_in = nn.Conv2d(1, c, **conv)
        self.conv_mid = nn.Conv2d(c, c, groups=c, **conv)
        self.lin_mid = nn.Conv2d(c, c, kernel_size=1)
        self.conv_end = nn.Conv2d(c, c, groups=c, **conv)
        self.lin_end = nn.Conv2d(c, c, kernel_size=1)
        self.lin_out = nn.Conv1d(N_MELS * c // 8, cfg.hidden_size, kernel_size=1)

    def forward(self, mel):                                           # [1, 128, F]
        h = F.silu(self.conv_in(mel.unsqueeze(1)))
        h = F.silu(self.lin_mid(self.conv_mid(h)))
        h = F.silu(self.lin_end(self.conv_end(h)))
        return self.lin_out(h.flatten(1, 2)).transpose(1, 2)          # [1, T, hidden]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = (xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.weight.type_as(x)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class EncAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        d = cfg.hidden_size
        self.n_heads, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = d // self.n_heads
        hd = self.head_dim
        self.q_proj = nn.Linear(d, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.out_proj = nn.Linear(self.n_heads * hd, d, bias=False)
        self.q_layernorm = RMSNorm(hd, cfg.norm_eps)
        self.k_layernorm = RMSNorm(hd, cfg.norm_eps)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        hd = self.head_dim
        q = self.q_layernorm(self.q_proj(x).view(B, T, self.n_heads, hd)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(x).view(B, T, self.n_kv, hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        att = torch.softmax((q @ k.transpose(-1, -2)) * hd ** -0.5 + mask, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, self.n_heads * hd)
        return self.out_proj(y)


class EncShortConv(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        d, L = cfg.hidden_size, cfg.conv_L_cache
        self.L = L
        self.B_proj = nn.Linear(d, d, bias=False)
        self.C_proj = nn.Linear(d, d, bias=False)
        self.x_proj = nn.Linear(d, d, bias=False)
        self.conv = nn.Conv1d(d, d, kernel_size=L, groups=d, padding=L - 1, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        T = x.size(1)
        Bx = (self.B_proj(x) * self.x_proj(x)).transpose(-1, -2)
        y = self.C_proj(x).transpose(-1, -2) * self.conv(Bx)[..., :T]
        return self.out_proj(y.transpose(-1, -2).contiguous())


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig, idx: int):
        super().__init__()
        self.is_attention = cfg.layer_types[idx] == "full_attention"
        if self.is_attention:
            self.self_attn = EncAttention(cfg)
        else:
            self.conv = EncShortConv(cfg)
        self.feed_forward = FeedForward(cfg.hidden_size, cfg.intermediate_size)
        self.operator_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)

    def forward(self, x, cos, sin, mask):
        h = self.operator_norm(x)
        h = self.self_attn(h, cos, sin, mask) if self.is_attention else self.conv(h)
        x = x + h
        return x + self.feed_forward(self.ffn_norm(x))


class AudioEncoder(nn.Module):
    """[1, 400 + 160k] waveform -> [1, ceil((k + 1) / 8), text hidden]."""

    def __init__(self, hf: dict, mel_filters: torch.Tensor, max_tokens: int):
        super().__init__()
        cfg = EncoderConfig.from_hf(hf)
        self.cfg = cfg
        self.log_mel = LogMel(mel_filters)
        self.downsample = Downsampler(cfg)
        self.layers = nn.ModuleList(EncLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.embedding_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        d = cfg.hidden_size
        self.proj_norm = nn.LayerNorm(d)
        self.proj_1 = nn.Linear(d, cfg.text_hidden)
        self.proj_2 = nn.Linear(cfg.text_hidden, cfg.text_hidden)

        hd = d // cfg.num_attention_heads
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
        freqs = torch.arange(max_tokens, dtype=torch.float)[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)

    def forward(self, wave):
        x = self.downsample(self.log_mel(wave))
        T = x.size(1)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        i = torch.arange(T).unsqueeze(1)
        j = torch.arange(T).unsqueeze(0)
        visible = (j <= i) & (i - j < self.cfg.sliding_window)
        mask = torch.where(visible, 0.0, float("-inf"))
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        x = self.embedding_norm(x)
        return self.proj_2(F.gelu(self.proj_1(self.proj_norm(x))))


def convert(hf_state: dict, cfg: EncoderConfig) -> dict:
    """Checkpoint `encoder.*` / `mlp.model.*` -> AudioEncoder."""
    d = cfg.hidden_size
    proj = {"mlp.model.0.": "proj_norm.", "mlp.model.1.": "proj_1.", "mlp.model.3.": "proj_2."}
    out = {}
    for name, t in hf_state.items():
        for src, dst in proj.items():
            if name.startswith(src):
                out[dst + name[len(src):]] = t
        if name.startswith("encoder.downsample."):
            out[name[len("encoder."):]] = t
        elif name.startswith("encoder.lfm.lfm."):
            key = name[len("encoder.lfm.lfm."):]
            if key.endswith("conv.in_proj.weight"):
                base = key[: -len("in_proj.weight")]
                b, c, x = t.split(d, dim=0)
                out[base + "B_proj.weight"], out[base + "C_proj.weight"], out[base + "x_proj.weight"] = b, c, x
            else:
                out[key] = t
    return out
