"""Exportable LFM3 text decoder.

A re-implementation of the HF `Lfm3ForCausalLM` (LiquidAI/LFM3-1.7B,
`modeling_lfm3.py`) shaped for ExecuTorch: batch 1, static KV caches and
short-conv states held as mutable buffers, and the attention kernel picked per
backend (`attn_impl`):

- "eager": plain torch attention over the cache. Reference / debugging only.
- "llama": `llama::update_cache` + `llama::custom_sdpa` (XNNPACK, Vulkan).
- "mlx":   `mlx::kv_cache_update` + `mlx::custom_sdpa` (MLX).

What LFM3 adds over LFM2 (and why ExecuTorch's llama Transformer cannot express it):
- three layer types: `conv` (gated short conv, kernel 7), `full_attention`
  and `sliding_attention` (256-token window), each attention type with its
  own RoPE base;
- zero-tail partial RoPE on full layers: the first `partial_rotary_factor *
  head_dim / 2` frequencies follow the full-head progression and the rest are
  zero, so `rotate_half` still pairs channel i with i + head_dim/2;
- a per-head sigmoid output gate (`gate_proj: hidden -> n_heads`);
- sandwich norm (`post_operator_norm`, `post_ffn_norm`).

The sliding-window mask depends only on (start_pos, T), so it is built once per
forward and shared by every sliding layer.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Lfm3TextConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    layer_types: list
    sliding_window: int
    conv_L_cache: int
    rope_parameters: dict
    norm_eps: float = 1e-5
    conv_bias: bool = False
    use_sandwich_norm: bool = True
    tie_word_embeddings: bool = True
    # Runtime shape, not part of the HF config.
    max_context_len: int = 4096
    attn_impl: str = "llama"
    # "llama" only: sliding layers read just the cache rows inside the window
    # (O(window) per step instead of O(context)). Off for Vulkan, whose SDPA
    # pattern needs the whole cache tensor as the custom_sdpa operand.
    window_slice: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_hf(cls, hf: dict, **runtime) -> "Lfm3TextConfig":
        # The text model sits under text_config (VL) or lfm (Audio).
        hf = hf.get("text_config") or hf.get("lfm") or hf
        names = {f for f in cls.__dataclass_fields__ if f not in ("extra",)}
        kwargs = {k: hf[k] for k in names if k in hf}
        kwargs.update(runtime)
        return cls(**kwargs)


class RMSNorm(nn.Module):
    # Same formulation as executorch's llama RMSNorm, so backend RMSNorm
    # patterns (Vulkan) match.
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = (xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.weight.type_as(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rope_tables(cfg: Lfm3TextConfig, layer_type: str):
    """cos/sin tables [max_context_len, head_dim], bit-identical to HF's
    `Lfm3RotaryEmbedding` (fp32 position * inv_freq, zero-tail partial RoPE)."""
    p = cfg.rope_parameters[layer_type]
    assert p.get("rope_type", "default") == "default", p
    dim = cfg.head_dim
    angles = int(p.get("partial_rotary_factor", 1.0) * dim) // 2
    inv_freq = 1.0 / (p["rope_theta"] ** (torch.arange(0, 2 * angles, 2, dtype=torch.float) / dim))
    inv_freq = torch.cat((inv_freq, torch.zeros(dim // 2 - angles, dtype=torch.float)))
    pos = torch.arange(cfg.max_context_len, dtype=torch.float)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Lfm3TextConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scaling = self.head_dim ** -0.5
        self.layer_type = cfg.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        hd = self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(hd, cfg.norm_eps)
        self.k_norm = RMSNorm(hd, cfg.norm_eps)
        self.gate_proj = nn.Linear(cfg.hidden_size, self.n_heads, bias=False)

        ctx = cfg.max_context_len
        if cfg.attn_impl == "mlx":
            from executorch.backends.mlx.llm.cache import KVCache as MLXKVCache

            self.kv_cache = MLXKVCache(1, ctx, self.n_kv, hd, enable_dynamic_shape=True)
        else:
            # llama::update_cache / custom_sdpa take BSHD caches.
            shape = (1, ctx, self.n_kv, hd)
            self.register_buffer("k_cache", torch.zeros(shape), persistent=False)
            self.register_buffer("v_cache", torch.zeros(shape), persistent=False)

    def forward(self, x, cos, sin, start_pos, window_mask):
        B, T, _ = x.shape
        hd = self.head_dim
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_heads, hd))
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv, hd))
        v = self.v_proj(x).view(B, T, self.n_kv, hd)

        # RoPE in BSHD: cos/sin are [T, hd] -> broadcast over heads.
        c, s = cos[:, None, :], sin[:, None, :]
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s

        impl = self.cfg.attn_impl
        mask = window_mask if self.is_sliding else None
        if impl == "llama":
            torch.ops.llama.update_cache(k, self.k_cache, start_pos)
            torch.ops.llama.update_cache(v, self.v_cache, start_pos)
            k_cache, v_cache = self.k_cache, self.v_cache
            if mask is not None and self.cfg.window_slice:
                mask, lo = mask
                k_cache = k_cache.narrow(1, lo, mask.size(1))
                v_cache = v_cache.narrow(1, lo, mask.size(1))
            # Positional: the op schema spells the dropout arg `drpout_p`.
            y = torch.ops.llama.custom_sdpa(
                q, k_cache, v_cache, start_pos, mask, 0.0, mask is None, None
            )
        elif impl == "mlx":
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            k_cache, v_cache = self.kv_cache.update(start_pos, k, v)
            if mask is None:
                y = torch.ops.mlx.custom_sdpa(
                    q, k_cache, v_cache, start_pos=start_pos, attn_mask=None,
                    dropout_p=0.0, is_causal=True, scale=self.scaling,
                )
            else:
                # custom_sdpa attends over cache[:start_pos + T]; aim it at the
                # whole cache and let the window mask block unwritten slots.
                ctx = self.cfg.max_context_len
                y = torch.ops.mlx.custom_sdpa(
                    q, k_cache, v_cache, start_pos=ctx - T, attn_mask=mask[None, None].to(q.dtype),
                    dropout_p=0.0, is_causal=False, scale=self.scaling,
                )
            y = y.transpose(1, 2)
        else:  # eager
            self.k_cache[:, start_pos : start_pos + T] = k
            self.v_cache[:, start_pos : start_pos + T] = v
            kk = self.k_cache[:, : start_pos + T].transpose(1, 2)
            vv = self.v_cache[:, : start_pos + T].transpose(1, 2)
            rep = self.n_heads // self.n_kv
            kk = kk.repeat_interleave(rep, dim=1)
            vv = vv.repeat_interleave(rep, dim=1)
            if mask is None:
                qpos = start_pos + torch.arange(T)[:, None]
                kpos = torch.arange(start_pos + T)[None, :]
                mask = torch.where(kpos <= qpos, 0.0, float("-inf"))
            else:
                mask = mask[:, : start_pos + T]
            att = (q.transpose(1, 2) @ kk.transpose(-1, -2)) * self.scaling + mask.to(q.dtype)
            y = (att.softmax(-1, dtype=torch.float32).to(q.dtype) @ vv).transpose(1, 2)

        gate = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        y = (y * gate).reshape(B, T, self.n_heads * hd)
        return self.o_proj(y)


class ShortConv(nn.Module):
    """Gated short conv. HF's fused `in_proj` (3*D rows: B, C, x) is split into
    three linears, which executorch's LFM2 port found faster on device."""

    def __init__(self, cfg: Lfm3TextConfig):
        super().__init__()
        d, L = cfg.hidden_size, cfg.conv_L_cache
        self.L = L
        self.B_proj = nn.Linear(d, d, bias=cfg.conv_bias)
        self.C_proj = nn.Linear(d, d, bias=cfg.conv_bias)
        self.x_proj = nn.Linear(d, d, bias=cfg.conv_bias)
        self.conv = nn.Conv1d(d, d, kernel_size=L, groups=d, padding=0, bias=cfg.conv_bias)
        self.out_proj = nn.Linear(d, d, bias=cfg.conv_bias)
        self.register_buffer("conv_state", torch.zeros(1, d, L - 1), persistent=False)

    def forward(self, x):
        T = x.size(1)
        Bg = self.B_proj(x).transpose(-1, -2)
        Cg = self.C_proj(x).transpose(-1, -2)
        xx = self.x_proj(x).transpose(-1, -2)
        Bx = torch.cat([self.conv_state, Bg * xx], dim=-1)
        with torch.no_grad():
            self.conv_state.copy_(Bx[..., -(self.L - 1):])
        y = Cg * self.conv(Bx)[..., :T]
        return self.out_proj(y.transpose(-1, -2).contiguous())


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Lfm3TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = cfg.layer_types[layer_idx]
        if self.layer_type == "conv":
            self.conv = ShortConv(cfg)
        else:
            self.self_attn = Attention(cfg, layer_idx)
        self.feed_forward = FeedForward(cfg.hidden_size, cfg.intermediate_size)
        self.operator_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.sandwich = cfg.use_sandwich_norm
        if self.sandwich:
            self.post_operator_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
            self.post_ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)

    def forward(self, x, rope, start_pos, window_mask):
        h = self.operator_norm(x)
        if self.layer_type == "conv":
            h = self.conv(h)
        else:
            cos, sin = rope[self.layer_type]
            h = self.self_attn(h, cos, sin, start_pos, window_mask)
        if self.sandwich:
            h = self.post_operator_norm(h)
        x = x + h
        f = self.feed_forward(self.ffn_norm(x))
        if self.sandwich:
            f = self.post_ffn_norm(f)
        return x + f


class Lfm3TextModel(nn.Module):
    """Decoder trunk: embeddings in, final-normed hidden states out.

    `forward(tokens, input_pos)` is the text-LLM contract (last-token logits);
    multimodal wrappers call `decode(embeds, input_pos)` directly.
    """

    def __init__(self, cfg: Lfm3TextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.embedding_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rope_types = sorted({t for t in cfg.layer_types if t != "conv"})
        for t in self.rope_types:
            cos, sin = rope_tables(cfg, t)
            self.register_buffer(f"{t}_cos", cos, persistent=False)
            self.register_buffer(f"{t}_sin", sin, persistent=False)

    def _window_mask(self, start_pos, T):
        """Additive mask: key j is visible to query i iff j <= i and
        i - j < sliding_window (HF's sliding causal mask).

        Covers the whole cache ([T, max_context_len]) by default. With
        `window_slice` it covers only keys [lo, start_pos + T), where lo is the
        oldest key any query in the chunk can see, and returns (mask, lo)."""
        W = self.cfg.sliding_window
        lo = 0
        kv_len = self.cfg.max_context_len
        if self.cfg.window_slice and self.cfg.attn_impl == "llama":
            lo = torch.sym_max(start_pos - W + 1, 0)
            kv_len = start_pos + T - lo
            torch._check(kv_len >= 1)
            torch._check(kv_len <= W + T - 1)
        qpos = start_pos + torch.arange(T, dtype=torch.long).unsqueeze(1)
        kpos = lo + torch.arange(kv_len, dtype=torch.long).unsqueeze(0)
        delta = qpos - kpos
        visible = (delta >= 0) & (delta < W)
        zero = torch.zeros(1, dtype=torch.float32)
        neg_inf = torch.full((1,), float("-inf"), dtype=torch.float32)
        mask = torch.where(visible, zero, neg_inf)
        if self.cfg.window_slice and self.cfg.attn_impl == "llama":
            return mask, lo
        return mask

    def decode(self, h, input_pos):
        T = h.size(1)
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        torch._check(start_pos + T <= self.cfg.max_context_len)
        rope = {}
        for t in self.rope_types:
            rope[t] = (
                getattr(self, f"{t}_cos").narrow(0, start_pos, T),
                getattr(self, f"{t}_sin").narrow(0, start_pos, T),
            )
        window_mask = None
        if "sliding_attention" in self.rope_types:
            window_mask = self._window_mask(start_pos, T)
        for layer in self.layers:
            h = layer(h, rope, start_pos, window_mask)
        return self.embedding_norm(h)

    def forward(self, tokens, input_pos):
        h = self.decode(self.embed_tokens(tokens), input_pos)
        # RNE's LLM spec takes float32 logits whatever the compute dtype.
        return self.lm_head(h[:, -1, :]).float()
