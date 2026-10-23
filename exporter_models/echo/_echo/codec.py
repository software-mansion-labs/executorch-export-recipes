"""The Higgs Audio v2 codec around Echo: decoder (codes -> audio) and encoder
(reference audio -> codes), spelled so each lowers as few delegate blobs as
the backends allow.

  decode : codes i64[C, T] -> audio f32[T * 960]          (24 kHz, 25 frames/s)
  encode : wav f32[N] (24 kHz, N a multiple of 960) -> codes i64[C, N / 960]
"""

import importlib.machinery
import math
import sys
import types

import torch
import torch.nn.functional as F
from torch import nn


def _sinc_kernel(orig, new, width=6, rolloff=0.99, dtype=torch.float32):
    base = min(orig, new) * rolloff
    w = math.ceil(width * orig / base)
    idx = torch.arange(-w, w + orig, dtype=torch.float64)[None, None] / orig
    t = torch.arange(0, -new, -1, dtype=torch.float64)[:, None, None] / new + idx
    t = (t * base).clamp(-width, width)
    window = torch.cos(t * math.pi / width / 2) ** 2
    t = t * math.pi
    k = torch.where(t == 0, torch.tensor(1.0, dtype=torch.float64), t.sin() / t)
    return (k * window * base / orig).to(dtype), w


def resample(waveform, orig_freq, new_freq, **_):
    """torchaudio.functional.resample's default (sinc_interp_hann, width 6,
    rolloff 0.99), ported: the encoder's 24 kHz -> 16 kHz step."""
    if orig_freq == new_freq:
        return waveform
    gcd = math.gcd(int(orig_freq), int(new_freq))
    orig, new = int(orig_freq) // gcd, int(new_freq) // gcd
    kernel, width = _sinc_kernel(orig, new, dtype=waveform.dtype)
    shape = waveform.shape
    x = waveform.reshape(-1, shape[-1])
    n = x.shape[-1]
    y = F.conv1d(F.pad(x[:, None], (width, width + orig)), kernel, stride=orig)
    y = y.transpose(1, 2).reshape(x.shape[0], -1)
    target = math.ceil(new * n / orig)
    return y[..., :target].reshape(shape[:-1] + (target,))


def _stub_torchaudio():
    """transformers' codec module imports torchaudio unconditionally, and no
    torchaudio wheel matches torch 2.14. Its only use is the resample above."""
    try:
        import torchaudio  # noqa: F401

        return
    except ImportError:
        pass
    ta = types.ModuleType("torchaudio")
    ta.__version__ = "0.0-stub"
    ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    ta.functional = types.ModuleType("torchaudio.functional")
    ta.functional.__spec__ = importlib.machinery.ModuleSpec("torchaudio.functional", None)
    ta.functional.resample = resample
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = ta.functional

    import transformers.utils.import_utils as iu

    iu.BACKENDS_MAPPING["torchaudio"] = (lambda: True, iu.BACKENDS_MAPPING["torchaudio"][1])


def load_codec(path: str):
    _stub_torchaudio()
    from transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer import (
        HiggsAudioV2TokenizerModel,
    )

    return HiggsAudioV2TokenizerModel.from_pretrained(path).eval()


class Snake(nn.Module):
    """DAC's Snake1d with 1/alpha baked in: the original recomputes
    (alpha + 1e-9).reciprocal() from the parameter on every call, which no
    partitioner folds, so each of the 36 activations split the delegate."""

    def __init__(self, s):
        super().__init__()
        self.register_buffer("alpha", s.alpha.detach().clone())
        self.register_buffer("inv", (s.alpha.detach() + 1e-9).reciprocal())

    def forward(self, x):
        return x + self.inv * torch.sin(self.alpha * x).pow(2)


def _swap_snake(m):
    for name, child in m.named_children():
        if type(child).__name__ == "Snake1d":
            setattr(m, name, Snake(child))
        else:
            _swap_snake(child)


class Decoder(nn.Module):
    """codec.decode() for one clip: codes [C, T] -> audio [T * 960]."""

    def __init__(self, codec):
        super().__init__()
        _swap_snake(codec.acoustic_decoder)
        self.q = codec.quantizer
        self.fc2 = codec.fc2
        self.dec = codec.acoustic_decoder

    def forward(self, codes):
        qs = self.q.quantizers
        x = qs[0].project_out(qs[0].codebook.decode(codes[0]))  # [T, 1024]
        for i in range(1, len(qs)):
            x = x + qs[i].project_out(qs[i].codebook.decode(codes[i]))
        # [T, 256] -> [1, 256, T] as a permute + view(-1): unsqueeze and strided
        # copies split the XNNPACK delegate.
        x = self.fc2(x).view(1, -1, 256).permute(0, 2, 1)
        return self.dec(x).reshape(-1)


class Encoder(nn.Module):
    """codec.encode() for one mono clip whose length is a whole number of frames."""

    def __init__(self, codec):
        super().__init__()
        # Plain bmm/softmax: SDPA's masked-row guard ops are not delegated.
        codec.semantic_model.config._attn_implementation = "eager"
        self.c = codec

    def forward(self, wav):  # [N]
        c, cfg = self.c, self.c.config
        x16 = resample(wav.view(1, -1), cfg.sample_rate, cfg.semantic_sample_rate)
        x16 = F.pad(x16, (160, 160))
        hs = c.semantic_model(x16, output_hidden_states=True).hidden_states
        sem = torch.stack(hs, dim=1).mean(dim=1)
        if cfg.semantic_downsample_factor > 1:
            sem = sem[:, :: cfg.semantic_downsample_factor, :]
        e_sem = c.encoder_semantic(sem.transpose(1, 2))
        x = wav.view(1, 1, -1)
        if c._get_conv1d_output_lengths(x.shape[2], c.acoustic_encoder) != e_sem.shape[2]:
            x = F.pad(x, (c.pad, c.pad))
        e_ac = c.acoustic_encoder(x)
        emb = c.fc(torch.cat([e_ac, e_sem], dim=1).transpose(1, 2)).transpose(1, 2)
        return c.quantizer.encode(emb, cfg.target_bandwidths[-1])[:, 0]  # [C, T]


def export_decoder(codec, t_static: int | None = None, t_max: int = 1024):
    """ExportedProgram of `decode`; dynamic T in [4, t_max] unless t_static."""
    dec = Decoder(codec).eval()
    t = t_static or 120
    ex = (torch.randint(0, 1024, (8, t)),)
    ref = codec.decode(ex[0][None]).audio_values.reshape(-1)
    assert (dec(*ex) - ref).abs().max() < 1e-4, "decoder wrapper diverges from codec.decode"
    ds = None if t_static else ({1: torch.export.Dim("T", min=4, max=t_max)},)
    return torch.export.export(dec, ex, dynamic_shapes=ds, strict=False)


def export_encoder(codec, seconds: float = 10.0):
    """ExportedProgram of `encode` at a fixed `seconds` of 24 kHz audio."""
    enc = Encoder(codec).eval()
    hop = codec.config.hop_length
    n = int(seconds * codec.config.sample_rate) // hop * hop
    g = torch.Generator().manual_seed(0)
    wav = torch.randn(n, generator=g) * 0.05
    ref = codec.encode(wav.view(1, 1, -1)).audio_codes[0]
    assert torch.equal(enc(wav), ref), "encoder wrapper diverges from codec.encode"
    return torch.export.export(enc, (wav,), strict=False)
