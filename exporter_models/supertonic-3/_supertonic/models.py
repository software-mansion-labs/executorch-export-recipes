"""PyTorch models replicating the four Supertonic 3 ONNX graphs.

Module/parameter names mirror the ONNX initializer names (with the
``tts.`` / ``vector_estimator.tts.ttl.`` prefixes stripped), so state
dicts produced by ``convert.py`` load directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .modules import (
    AttnEncoder,
    ChannelLayerNorm,
    ConvNeXtBlock,
    ConvNeXtStack,
    LinearNet,
    NetConv1d,
    RotaryCrossAttention,
    TanhAttention,
    TextEmbedder,
    TimeEncoder,
)

N_CHARS = 8322


# ---------------------------------------------------------------------------
# Text encoder (text_encoder.onnx)
# ---------------------------------------------------------------------------


class TTLTextEncoder(nn.Module):
    def __init__(self, dim=256, intermediate_dim=1024, n_heads=4, n_layers=4,
                 conv_dilations=(1, 1, 2, 2, 4, 4), ksz=5):
        super().__init__()
        self.text_embedder = TextEmbedder(N_CHARS, dim)
        self.convnext = ConvNeXtStack(dim, intermediate_dim, ksz, conv_dilations)
        self.attn_encoder = AttnEncoder(dim, intermediate_dim, n_heads, n_layers)

    def forward(self, text_ids, text_mask):
        x = self.text_embedder(text_ids, text_mask)
        conv = self.convnext(x, text_mask)
        attn = self.attn_encoder(conv, text_mask)
        return (attn + conv) * text_mask


class SpeechPromptedTextEncoder(nn.Module):
    def __init__(self, dim=256, n_heads=2):
        super().__init__()
        self.attention1 = TanhAttention(dim, dim, dim, dim, n_heads)
        self.attention2 = TanhAttention(dim, dim, dim, dim, n_heads)
        self.norm = ChannelLayerNorm(dim)

    def forward(self, text, style_keys, style_values, text_mask):
        # text: [B, C, T]; style_keys/values: [B, 50, C]; text_mask: [B, 1, T]
        xt = text.transpose(1, 2)
        mt = text_mask.transpose(1, 2)
        y = self.attention1(xt, style_keys, style_values, query_mask=mt) * mt + xt
        y = self.attention2(y, style_keys, style_values, query_mask=mt) * mt + xt
        y = self.norm.norm(y)  # LayerNorm over last dim of [B, T, C]
        return y.transpose(1, 2) * text_mask


class _StyleTokenLayer(nn.Module):
    """Holds the learned style-key tokens (keys of the style attention)."""

    def __init__(self, n_style=50, dim=256):
        super().__init__()
        self.style_key = nn.Parameter(torch.zeros(1, n_style, dim))


class _StyleEncoder(nn.Module):
    def __init__(self, n_style=50, dim=256):
        super().__init__()
        self.style_token_layer = _StyleTokenLayer(n_style, dim)


class TextEncoderModel(nn.Module):
    """text_ids + style_ttl + text_mask -> text_emb [B, 256, T]."""

    def __init__(self):
        super().__init__()
        self.text_encoder = TTLTextEncoder()
        self.style_encoder = _StyleEncoder()
        self.speech_prompted_text_encoder = SpeechPromptedTextEncoder()

    def forward(self, text_ids, style_ttl, text_mask):
        x = self.text_encoder(text_ids, text_mask)
        keys = self.style_encoder.style_token_layer.style_key.expand(
            x.shape[0], -1, -1
        )
        return self.speech_prompted_text_encoder(x, keys, style_ttl, text_mask)


# ---------------------------------------------------------------------------
# Duration predictor (duration_predictor.onnx)
# ---------------------------------------------------------------------------


class SentenceEncoder(nn.Module):
    def __init__(self, dim=64, intermediate_dim=256, n_heads=2, n_layers=2,
                 n_conv_layers=6, ksz=5):
        super().__init__()
        self.sentence_token = nn.Parameter(torch.zeros(1, dim, 1))
        self.text_embedder = TextEmbedder(N_CHARS, dim)
        self.convnext = ConvNeXtStack(dim, intermediate_dim, ksz, [1] * n_conv_layers)
        self.attn_encoder = AttnEncoder(dim, intermediate_dim, n_heads, n_layers)
        self.proj_out = NetConv1d(dim, dim, 1, bias=False)

    def forward(self, text_ids, text_mask):
        x = self.text_embedder(text_ids, text_mask)
        bsz = x.shape[0]
        x = torch.cat([self.sentence_token.expand(bsz, -1, -1), x], dim=-1)
        mask = torch.cat(
            [torch.ones_like(text_mask[:, :, :1]), text_mask], dim=-1
        )
        conv = self.convnext(x, mask)
        attn = self.attn_encoder(conv, mask)
        x = attn + conv
        # proj_out is a 1x1 conv over a length-1 slice; compute it as the
        # equivalent linear so the XNNPACK channels-last pass does not choke on
        # a conv feeding the downstream flatten. Numerically identical.
        w = self.proj_out.net.weight.squeeze(-1)  # [dim, dim]
        sent = F.linear(x[:, :, 0], w).unsqueeze(-1) * mask[:, :, :1]
        return sent  # [B, dim, 1]


class _DurationHead(nn.Module):
    def __init__(self, in_dim=192, hdim=128):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, hdim), nn.Linear(hdim, 1)])
        self.activation = nn.PReLU(1)

    def forward(self, h):
        h = self.layers[0](h)
        w = self.activation.weight  # manual PReLU (XNNPACK-friendly)
        h = torch.where(h >= 0, h, w * h)
        return self.layers[1](h)


class DurationPredictorModel(nn.Module):
    """text_ids + style_dp + text_mask -> duration in seconds [B]."""

    def __init__(self):
        super().__init__()
        self.sentence_encoder = SentenceEncoder()
        self.predictor = _DurationHead()

    def forward(self, text_ids, style_dp, text_mask):
        bsz = text_ids.shape[0]
        sent = self.sentence_encoder(text_ids, text_mask).reshape(bsz, -1)
        h = torch.cat([sent, style_dp.reshape(bsz, -1)], dim=1)
        return torch.exp(self.predictor(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Vector field / estimator (vector_estimator.onnx)
# ---------------------------------------------------------------------------


class _TimeCond(nn.Module):
    def __init__(self, idim=512, time_dim=64):
        super().__init__()
        self.linear = LinearNet(time_dim, idim)

    def forward(self, temb):
        return self.linear(temb)


class _TextCondBlock(nn.Module):
    def __init__(self, dim=512, text_dim=256, n_heads=8):
        super().__init__()
        self.attn = RotaryCrossAttention(dim, text_dim, n_heads)
        self.norm = ChannelLayerNorm(dim)

    def forward(self, x, text_t, text_mask, latent_mask):
        xt = (x * latent_mask).transpose(1, 2)
        y = self.attn(xt, text_t, text_mask, latent_mask)
        y = y * latent_mask.transpose(1, 2)
        return self.norm(x * latent_mask + y.transpose(1, 2)) * latent_mask


class _StyleCondBlock(nn.Module):
    def __init__(self, dim=512, style_dim=256, n_units=256, n_heads=2):
        super().__init__()
        self.attention = TanhAttention(dim, style_dim, n_units, dim, n_heads)
        self.norm = ChannelLayerNorm(dim)

    def forward(self, x, style_keys, style_values, latent_mask):
        mt = latent_mask.transpose(1, 2)
        xt = (x * latent_mask).transpose(1, 2)
        y = self.attention(xt, style_keys, style_values, query_mask=mt) * mt
        return self.norm(x * latent_mask + y.transpose(1, 2)) * latent_mask


class VectorField(nn.Module):
    def __init__(self, dim=512, latent_dim=144, text_dim=256, n_blocks=4,
                 intermediate_dim=2048, ksz=5):
        super().__init__()
        self.proj_in = NetConv1d(latent_dim, dim, 1, bias=False)
        self.time_encoder = TimeEncoder(64, 256)
        blocks = []
        for _ in range(n_blocks):
            blocks += [
                ConvNeXtStack(dim, intermediate_dim, ksz, [1, 2, 4, 8]),
                _TimeCond(dim, 64),
                ConvNeXtStack(dim, intermediate_dim, ksz, [1]),
                _TextCondBlock(dim, text_dim),
                ConvNeXtStack(dim, intermediate_dim, ksz, [1]),
                _StyleCondBlock(dim, text_dim),
            ]
        self.main_blocks = nn.ModuleList(blocks)
        self.last_convnext = ConvNeXtStack(dim, intermediate_dim, ksz, [1, 1, 1, 1])
        self.proj_out = NetConv1d(dim, latent_dim, 1, bias=False)

    def forward(self, x, text, style_keys, style_values, text_mask, latent_mask, t):
        h = self.proj_in(x) * latent_mask
        temb = self.time_encoder(t)
        text_t = text.transpose(1, 2)
        for g in range(len(self.main_blocks) // 6):
            b = 6 * g
            h = self.main_blocks[b](h, latent_mask)
            h = (h + self.main_blocks[b + 1](temb).unsqueeze(-1)) * latent_mask
            h = self.main_blocks[b + 2](h, latent_mask)
            h = self.main_blocks[b + 3](h, text_t, text_mask, latent_mask)
            h = self.main_blocks[b + 4](h, latent_mask)
            h = self.main_blocks[b + 5](h, style_keys, style_values, latent_mask)
        h = self.last_convnext(h, latent_mask)
        return self.proj_out(h) * latent_mask


class _UncondMasker(nn.Module):
    def __init__(self, text_dim=256, n_style=50, style_dim=256):
        super().__init__()
        self.text_special_token = nn.Parameter(torch.zeros(1, text_dim, 1))
        self.style_key_special_token = nn.Parameter(torch.zeros(1, n_style, style_dim))
        self.style_value_special_token = nn.Parameter(
            torch.zeros(1, n_style, style_dim)
        )


class VectorEstimatorModel(nn.Module):
    """One Euler step of the CFG flow-matching ODE.

    denoised = (x_t + dt * (cfg_a * v_cond - cfg_b * v_uncond)) * mask
    with cfg_a=4, cfg_b=3 (guidance scale 3), dt = 1/total_step.
    """

    CFG_A = 4.0
    CFG_B = 3.0

    def __init__(self):
        super().__init__()
        self.vector_field = VectorField()
        self.uncond_masker = _UncondMasker()
        self.register_buffer("style_key", torch.zeros(1, 50, 256))

    def forward(self, noisy_latent, text_emb, style_ttl, text_mask, latent_mask,
                current_step, total_step):
        bsz, _, text_len = text_emb.shape
        t = (current_step / total_step).repeat(2)
        dt = (1.0 / total_step).view(-1, 1, 1)

        x2 = noisy_latent.repeat(2, 1, 1)
        lm2 = latent_mask.repeat(2, 1, 1)
        tm2 = text_mask.repeat(2, 1, 1)
        text2 = torch.cat(
            [
                text_emb,
                self.uncond_masker.text_special_token.expand(bsz, -1, text_len),
            ],
            dim=0,
        )
        style_v2 = torch.cat(
            [
                style_ttl,
                self.uncond_masker.style_value_special_token.expand(bsz, -1, -1),
            ],
            dim=0,
        )
        style_k2 = torch.cat(
            [
                self.style_key.expand(bsz, -1, -1),
                self.uncond_masker.style_key_special_token.expand(bsz, -1, -1),
            ],
            dim=0,
        )

        v = self.vector_field(x2, text2, style_k2, style_v2, tm2, lm2, t)
        v_cond, v_uncond = v[:bsz], v[bsz:]
        v = self.CFG_A * v_cond - self.CFG_B * v_uncond
        return (noisy_latent + dt * v) * latent_mask


# ---------------------------------------------------------------------------
# Vocoder (vocoder.onnx) — the autoencoder decoder
# ---------------------------------------------------------------------------


class _BatchNormWrap(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.norm(x)


class _VocoderHead(nn.Module):
    def __init__(self, idim=512, hdim=2048, odim=512, ksz=3):
        super().__init__()
        self.layer1 = NetConv1d(idim, hdim, ksz, causal=True)
        self.act = nn.PReLU(1)
        self.layer2 = nn.Conv1d(hdim, odim, 1, bias=False)

    def forward(self, x):
        h = self.layer1(x)
        w = self.act.weight  # manual PReLU (XNNPACK-friendly)
        h = torch.where(h >= 0, h, w * h)
        return self.layer2(h)


class _AEDecoder(nn.Module):
    def __init__(self, ldim=24, hdim=512, intermediate_dim=2048, ksz=7,
                 dilations=(1, 2, 4, 1, 2, 4, 1, 1, 1, 1)):
        super().__init__()
        self.embed = NetConv1d(ldim, hdim, ksz, causal=True)
        self.convnext = nn.ModuleList(
            ConvNeXtBlock(hdim, intermediate_dim, ksz, d, flat_dwconv=False,
                          causal=True)
            for d in dilations
        )
        self.final_norm = _BatchNormWrap(hdim)
        self.head = _VocoderHead(hdim, intermediate_dim, hdim)

    def forward(self, x):
        h = self.embed(x)
        for block in self.convnext:
            h = block(h)
        h = self.final_norm(h)
        return self.head(h)


class _AE(nn.Module):
    def __init__(self, ldim=24, hdim=512):
        super().__init__()
        self.register_buffer("latent_mean", torch.zeros(1, ldim, 1))
        self.register_buffer("latent_std", torch.ones(1, ldim, 1))
        self.decoder = _AEDecoder(ldim, hdim)


class VocoderModel(nn.Module):
    """latent [B, 144, L] -> waveform [B, L*6*512]."""

    def __init__(self, ldim=24, chunk_compress_factor=6):
        super().__init__()
        self.ldim = ldim
        self.ccf = chunk_compress_factor
        self.register_buffer("ttl_scale", torch.tensor(0.25))
        self.ae = _AE(ldim)

    def forward(self, latent):
        x = latent / self.ttl_scale
        bsz, _, length = x.shape
        # [B, ldim*ccf, L] -> [B, ldim, ccf, L] -> [B, ldim, L*ccf]
        x = x.view(bsz, self.ldim, self.ccf, length)
        x = x.permute(0, 1, 3, 2).reshape(bsz, self.ldim, length * self.ccf)
        x = x * self.ae.latent_std + self.ae.latent_mean
        h = self.ae.decoder(x)  # [B, 512, L*ccf]
        return h.transpose(1, 2).reshape(bsz, -1)
