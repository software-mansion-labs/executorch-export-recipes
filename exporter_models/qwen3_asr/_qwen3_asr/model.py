"""Export-friendly Qwen3-ASR (audio encoder + Qwen3 text decoder).

The audio tower is a port of `Qwen3ASRAudioEncoder` from the `qwen-asr`
package (Apache-2.0, `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`)
reshaped for torch.export. The text decoder is plain Qwen3 (the `thinker.model`
weights), built on executorch's llama transformer like the qwen3 exporter.

Methods the exporters build from these modules (D = 1024 for 0.6B):
  preprocess      : waveform f32[T]                      -> mel f32[1, 128, T // 160]
  audio_encoder   : (mel f32[1, 128, 100c], i64[1] valid) -> f32[1, 13c, D]
  token_embedding : i64[1, S]                            -> f32[1, S, D]
  text_decoder    : (f32[1, S, D], i64[1] start pos)     -> f32[1, vocab]
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SOURCE_REPO = "Qwen/Qwen3-ASR-0.6B"

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
# The tower convolves the mel in independent 100-frame (1 s) chunks, each giving
# 13 tokens, and attends within 8-chunk (8 s) windows (`n_window_infer`).
CHUNK_FRAMES = 100
AUDIO_TOKENS_PER_CHUNK = 13
WINDOW_FRAMES = 800
WINDOW_CHUNKS = WINDOW_FRAMES // CHUNK_FRAMES

# One transcription call covers at most 30 s of audio (390 audio tokens). With
# the ~20-token chat prompt that leaves >600 tokens of transcript in the context.
MAX_AUDIO_SAMPLES = 30 * SAMPLE_RATE
MAX_CONTEXT_LEN = 1024

_MASK_VALUE = -1e4  # finite so fp16 backends do not produce inf - inf


def audio_token_count(frames: int) -> int:
    """Number of audio tokens the tower emits for `frames` mel frames."""
    rem = frames % CHUNK_FRAMES
    tail = ((((rem - 1) // 2 + 1) - 1) // 2 + 1 - 1) // 2 + 1 if rem else 0
    return (frames // CHUNK_FRAMES) * AUDIO_TOKENS_PER_CHUNK + tail


def _slaney_mel_filters() -> np.ndarray:
    """Whisper's 128-bin mel filter bank, [n_fft // 2 + 1, n_mels]."""
    from transformers.audio_utils import mel_filter_bank

    return mel_filter_bank(
        num_frequency_bins=1 + N_FFT // 2,
        num_mel_filters=N_MELS,
        min_frequency=0.0,
        max_frequency=SAMPLE_RATE / 2,
        sampling_rate=SAMPLE_RATE,
        norm="slaney",
        mel_scale="slaney",
    )


class MelPreprocessor(nn.Module):
    """WhisperFeatureExtractor (128 bins) as an exportable graph.

    Matches `_torch_extract_fbank_features`: reflect-padded centred STFT with a
    periodic Hann window, power spectrum, slaney mel, log10, the last frame
    dropped, then the global `max - 8` floor and `(x + 4) / 4` scaling.
    """

    def __init__(self):
        super().__init__()
        half = N_FFT // 2
        n_freqs = half + 1
        window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float64)
        k = torch.arange(n_freqs, dtype=torch.float64).view(-1, 1)
        n = torch.arange(N_FFT, dtype=torch.float64).view(1, -1)
        cos = torch.cos(2 * math.pi * k * n / N_FFT) * window
        sin = -torch.sin(2 * math.pi * k * n / N_FFT) * window
        # Real and imaginary rows in two contiguous blocks so the power can be
        # computed by slicing, without a reduction no backend delegates.
        self.register_buffer("stft_kernel", torch.cat([cos, sin]).float().unsqueeze(1))
        self.register_buffer("mel_filters", torch.from_numpy(_slaney_mel_filters()).float().T.contiguous())
        # Reflection of the `half` samples next to each edge, as a matmul against
        # an anti-identity: flip() has no delegate on any backend.
        self.register_buffer("reverse", torch.eye(half).flip(0).contiguous())
        self.half = half
        self.n_freqs = n_freqs

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        h = self.half
        x = waveform.view(1, -1)
        left = x[:, 1 : h + 1] @ self.reverse
        right = x[:, -h - 1 : -1] @ self.reverse
        x = torch.cat([left, x, right], dim=1).view(1, 1, -1)
        stft = F.conv1d(x, self.stft_kernel, stride=HOP_LENGTH)
        real = stft[:, : self.n_freqs, :-1]
        imag = stft[:, self.n_freqs :, :-1]
        power = real * real + imag * imag
        mel = torch.matmul(self.mel_filters, power)  # [1, n_mels, frames]
        log_mel = torch.log10(torch.clamp(mel, min=1e-10))
        log_mel = torch.maximum(log_mel, log_mel.amax() - 8.0)
        return (log_mel + 4.0) / 4.0


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [N, D], mask: [1, N] additive over keys
        n, d = x.shape
        h = self.self_attn_layer_norm(x)
        # Heads folded into the batch dim and attention spelled as bmm/softmax/bmm:
        # SDPA's decomposition adds a fully-masked-row NaN guard (eq/any/
        # logical_not/where per layer) and a 4-D matmul lowers to expand_copy +
        # bmm, none of which any backend delegates. Same math either way.
        q = self.q_proj(h).view(n, self.n_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(h).view(n, self.n_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(h).view(n, self.n_heads, self.head_dim).transpose(0, 1)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.head_dim**-0.5 + mask
        attn = torch.bmm(torch.softmax(scores, dim=-1), v)  # [H, N, hd]
        x = x + self.out_proj(attn.transpose(0, 1).reshape(n, d))
        h = self.final_layer_norm(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class AudioEncoder(nn.Module):
    """The Qwen3-ASR audio tower over ONE attention window (up to 8 s).

    `mel` holds c whole 100-frame chunks; a partial last chunk is zero padded by
    the caller, exactly as the reference pads it. `n_valid` is the number of
    real tokens, so padding keys are masked out of attention (the reference
    drops them before attention). Output tokens past `n_valid` are garbage and
    the caller discards them. Windows are independent in the reference
    (block-diagonal attention over `n_window_infer` frames), so longer audio is
    encoded window by window and concatenated.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        hidden = cfg["downsample_hidden_size"]
        self.conv2d1 = nn.Conv2d(1, hidden, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(hidden, hidden, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(hidden, hidden, 3, 2, padding=1)
        freq = (((N_MELS + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(hidden * freq, d, bias=False)
        self.layers = nn.ModuleList(
            _EncoderLayer(d, cfg["encoder_attention_heads"], cfg["encoder_ffn_dim"])
            for _ in range(cfg["encoder_layers"])
        )
        self.ln_post = nn.LayerNorm(d)
        self.proj1 = nn.Linear(d, d)
        self.proj2 = nn.Linear(d, cfg["output_dim"])
        # SinusoidsPositionEmbedding, restarted at every chunk.
        channels = d
        inc = np.log(10000) / (channels // 2 - 1)
        inv = torch.exp(-inc * torch.arange(channels // 2).float())
        t = torch.arange(AUDIO_TOKENS_PER_CHUNK)[:, None].float() * inv[None, :]
        self.register_buffer("pos", torch.cat([torch.sin(t), torch.cos(t)], dim=1))
        self.register_buffer("token_index", torch.arange(WINDOW_CHUNKS * AUDIO_TOKENS_PER_CHUNK).float(), persistent=False)

    def forward(self, mel: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        c = mel.shape[2] // CHUNK_FRAMES
        x = mel.view(N_MELS, c, CHUNK_FRAMES).transpose(0, 1).unsqueeze(1)  # [c, 1, 128, 100]
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))  # [c, H, 16, 13]
        cc, ch, f, t = x.shape
        # conv_out runs on 2-D [c*13, H*16]: a 3-D linear with a dynamic batch
        # decomposes into a bmm that copies the weight once per chunk.
        x = self.conv_out(x.permute(0, 3, 1, 2).reshape(cc * t, ch * f))
        x = (x.view(cc, t, -1) + self.pos).view(cc * t, -1)
        # Additive key mask: 0 for real tokens, _MASK_VALUE for padding. Built
        # from a constant index row with elementwise ops, which every backend
        # takes, rather than arange/where, which they do not.
        idx = self.token_index[: x.shape[0]]
        mask = (torch.clamp(idx - n_valid.to(idx.dtype) + 1.0, 0.0, 1.0) * _MASK_VALUE).view(1, -1)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_post(x)
        return self.proj2(F.gelu(self.proj1(x))).unsqueeze(0)


class TokenEmbedding(nn.Module):
    def __init__(self, embedding: nn.Module):
        super().__init__()
        self.embedding = embedding

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)


class TextDecoder(nn.Module):
    """Embeddings in, last-position logits out; the KV cache lives in the graph."""

    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.text_model = text_model

    def forward(self, embeddings: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        return self.text_model(None, {"input_pos": input_pos}, embeddings)


_TEXT_FROM_HF = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "self_attn.q_norm.weight": "attention.q_norm_fn.weight",
    "self_attn.k_norm.weight": "attention.k_norm_fn.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    # executorch's FeedForward applies the activation to w1, Qwen to gate_proj.
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
}


def _text_state_dict(sd: dict) -> dict:
    out = {}
    for key, value in sd.items():
        if not key.startswith("thinker.") or key.startswith("thinker.audio_tower."):
            continue
        key = key[len("thinker.") :]
        if key in _TEXT_FROM_HF:
            out[_TEXT_FROM_HF[key]] = value
            continue
        parts = key.split(".")  # model.layers.<i>.<rest>
        out[f"layers.{parts[2]}.{_TEXT_FROM_HF['.'.join(parts[3:])]}"] = value
    return out


def _audio_state_dict(sd: dict) -> dict:
    prefix = "thinker.audio_tower."
    out = {}
    for key, value in sd.items():
        if not key.startswith(prefix):
            continue
        key = key[len(prefix) :].replace("self_attn.", "")
        out[key] = value
    return out


class Qwen3Asr:
    """Loads the checkpoint once and hands out the exportable modules."""

    def __init__(self, checkpoint_dir: str | None = None):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        path = Path(checkpoint_dir or snapshot_download(SOURCE_REPO, allow_patterns=["*.json", "*.safetensors"]))
        self.config = json.loads((path / "config.json").read_text())["thinker_config"]
        self.state_dict = {k: v.float() for k, v in load_file(str(path / "model.safetensors")).items()}
        self.text_config = self.config["text_config"]
        self.audio_config = self.config["audio_config"]

    def preprocessor(self) -> MelPreprocessor:
        return MelPreprocessor().eval()

    def audio_encoder(self) -> AudioEncoder:
        enc = AudioEncoder(self.audio_config)
        enc.load_state_dict(_audio_state_dict(self.state_dict), strict=False)
        missing = set(enc.state_dict()) - set(_audio_state_dict(self.state_dict)) - {"pos"}
        assert not missing, f"audio tower weights missing: {missing}"
        return enc.eval()

    def text_params(self) -> dict:
        t = self.text_config
        # Every audio and text position carries the same id in all three MRoPE
        # sections, so the interleaved MRoPE reduces to ordinary 1-D RoPE.
        return {
            "dim": t["hidden_size"],
            "hidden_dim": t["intermediate_size"],
            "n_heads": t["num_attention_heads"],
            "n_kv_heads": t["num_key_value_heads"],
            "head_dim": t["head_dim"],
            "n_layers": t["num_hidden_layers"],
            "norm_eps": t["rms_norm_eps"],
            "rope_theta": float(t["rope_theta"]),
            "vocab_size": t["vocab_size"],
            "ffn_dim_multiplier": 1,
            "use_scaled_rope": False,
            "use_hf_rope": True,
            "attention_qkv_bias": False,
            "use_qk_norm": True,
            "qk_norm_before_rope": True,
        }

    def text_model(self, **model_args) -> nn.Module:
        """executorch llama Transformer with the Qwen3 decoder weights loaded.

        `model_args` go straight to `ModelArgs` (kv cache, sdpa op, attention
        type, dynamic shape, context length); backend source transforms are
        applied by the caller.
        """
        from executorch.examples.models.llama.llama_transformer import construct_transformer
        from executorch.examples.models.llama.model_args import ModelArgs

        args = ModelArgs(max_batch_size=1, **self.text_params(), **model_args)
        model = construct_transformer(args)
        missing, unexpected = model.load_state_dict(_text_state_dict(self.state_dict), strict=False, assign=True)
        assert not unexpected, unexpected
        assert all("cache" in k or "mask" in k or "freqs" in k for k in missing), missing
        return model.eval()

    def token_embedding(self) -> nn.Embedding:
        weight = self.state_dict["thinker.model.embed_tokens.weight"]
        emb = nn.Embedding(weight.shape[0], weight.shape[1])
        emb.weight = nn.Parameter(weight.clone())
        return emb.eval()
