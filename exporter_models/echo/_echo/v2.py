"""Echo's v2 backbone contract: prefill / step / polish (MLX).

  prefill : (ids i64[NT], ref i64[8, R]) -> K, V  f16[24, 1, 6, P, 128]   (P = NT + R)
  step    : (tokens i64[8, T], K, V) -> confidence f32[2, T], prediction i64[2, T]
  polish  : (tokens i64[8, T], bias f16[1, 1, 1, P], K, V) -> tokens i64[6, T]

`step` is one main sampler pass, reduced in-graph to codebooks 0-1's best code
and its log-probability. `polish` runs both polish passes (bias +1 on the
reference keys) and returns codebooks 2-7: 2-3 from the first pass, 4-7 guided
by (second - first). The Gumbel top-k and remasking stay in the caller.

Per pass this moves two stacked fp16 caches instead of 48 fp32 tensors and
returns [2, T] instead of [8, T, 1024]. Q/K/V and gate/up are fused linears,
attention is aten SDPA (MLX SdpaNode, GQA native) and step/polish RoPE is
mlx::rope with its offset P read off the cache shape. prefill keeps a table
RoPE: mlx::rope takes a symbolic offset only, and prefill's is 0.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .model import Echo

NT_MAX = 512
R_MAX = 512
T_MAX = 1024
P_MAX = 1024


class _Layer(nn.Module):
    def __init__(self, layer, cfg):
        super().__init__()
        a = layer.self_attn
        self.H, self.KV, self.HD = a.n_heads, a.n_kv, a.hd
        self.ln1, self.ln2 = layer.input_layernorm, layer.post_attention_layernorm
        self.q_norm, self.k_norm = a.q_norm, a.k_norm
        self.qkv = nn.Linear(cfg["hidden_size"], (self.H + 2 * self.KV) * self.HD, bias=False)
        self.qkv.weight.data = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).clone()
        self.o = a.o_proj
        self.f = cfg["intermediate_size"]
        self.gate_up = nn.Linear(cfg["hidden_size"], 2 * self.f, bias=False)
        self.gate_up.weight.data = torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight]).clone()
        self.down = layer.mlp.down_proj

    def qkv_heads(self, h):
        """h [N, D] -> q [1, H, N, HD], k, v [1, KV, N, HD] (q/k normed, not rotated)."""
        qkv = self.qkv(h)
        q = qkv[:, : self.H * self.HD].view(-1, self.H, self.HD)
        k = qkv[:, self.H * self.HD : (self.H + self.KV) * self.HD].view(-1, self.KV, self.HD)
        v = qkv[:, (self.H + self.KV) * self.HD :].view(-1, self.KV, self.HD)
        q = self.q_norm(q).transpose(0, 1).unsqueeze(0)
        k = self.k_norm(k).transpose(0, 1).unsqueeze(0)
        return q, k, v.transpose(0, 1).unsqueeze(0)

    def finish(self, x, attn):
        x = x + self.o(attn.squeeze(0).transpose(0, 1).reshape(-1, self.H * self.HD))
        gu = self.gate_up(self.ln2(x))
        return x + self.down(F.silu(gu[:, : self.f]) * gu[:, self.f :])


class EchoV2(nn.Module):
    def __init__(self, m: Echo, mlx_rope: bool = True):
        super().__init__()
        cfg = m.cfg
        self.m = m
        self.mask_id, self.vocab = m.mask_id, m.vocab
        self.layers = nn.ModuleList(_Layer(layer, cfg) for layer in m.llm.layers)
        self.theta = float(cfg["rope_parameters"]["rope_theta"])
        self.HD = cfg["head_dim"]
        self.mlx_rope = mlx_rope
        self.register_buffer("offsets", (torch.arange(m.n_cb) * m.vocab)[:, None], persistent=False)
        w = m.audio_heads.weight.view(m.n_cb, m.vocab, -1)
        self.heads01 = nn.Parameter(w[:2].reshape(2 * m.vocab, -1).clone())
        self.heads27 = nn.Parameter(w[2:].reshape(6 * m.vocab, -1).clone())

    def embed_audio(self, codes):
        return self.m.audio_embeddings(codes + self.offsets).sum(0)

    def table_rope(self, x, pos):
        cos = F.embedding(pos, self.m.rope_cos).to(x.dtype)
        sin = F.embedding(pos, self.m.rope_sin).to(x.dtype)
        half = self.HD // 2
        return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin

    def rope(self, x, start, pos):
        if self.mlx_rope:
            return torch.ops.mlx.rope(x, self.HD, start, False, self.theta, 1.0, None)
        return self.table_rope(x, pos)

    def prefill(self, ids, ref):
        x = torch.cat([self.m.llm.embed_tokens(ids), self.embed_audio(ref)], dim=0)
        pos = torch.arange(x.shape[0])
        ks, vs = [], []
        for layer in self.layers:
            q, k, v = layer.qkv_heads(layer.ln1(x))
            q, k = self.table_rope(q, pos), self.table_rope(k, pos)
            x = layer.finish(x, F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
            ks.append(k)
            vs.append(v)
        return torch.stack(ks), torch.stack(vs)

    def hidden(self, tokens, K, V, mask):
        P = K.shape[3]
        pos = torch.arange(tokens.shape[1]) + P
        x = self.embed_audio(tokens)
        for i, layer in enumerate(self.layers):
            q, k, v = layer.qkv_heads(layer.ln1(x))
            q, k = self.rope(q, P, pos), self.rope(k, P, pos)
            kk, vv = torch.cat([K[i], k], dim=2), torch.cat([V[i], v], dim=2)
            x = layer.finish(x, F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, enable_gqa=True))
        return self.m.llm.norm(x)

    def log_probs(self, x, w, n):
        logits = F.linear(x, w).float().view(-1, n, self.vocab).transpose(0, 1)
        return torch.log_softmax(logits, dim=-1)[..., : self.mask_id]  # the mask id is never a prediction

    def step(self, tokens, K, V):
        lp = self.log_probs(self.hidden(tokens, K, V, None), self.heads01, 2)
        return lp.amax(-1), lp.argmax(-1)  # max.dim is not delegated to MLX

    def polish(self, tokens, bias, K, V):
        mask = torch.cat([bias, torch.zeros(1, 1, 1, tokens.shape[1], dtype=bias.dtype)], dim=-1)
        first = self.log_probs(self.hidden(tokens, K, V, mask), self.heads27, 6)  # codebooks 2-7
        a1 = first[:2].argmax(-1)
        tokens = torch.cat([tokens[:2], a1, tokens[4:]], dim=0)
        second = self.log_probs(self.hidden(tokens, K, V, mask), self.heads27[2 * self.vocab :], 4)  # 4-7
        return torch.cat([a1, (second + 0.5 * (second - first[2:])).argmax(-1)], dim=0)


class _Prefill(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, ref):
        return self.m.prefill(ids, ref)


class _Step(_Prefill):
    def forward(self, tokens, K, V):
        return self.m.step(tokens, K, V)


class _Polish(_Prefill):
    def forward(self, tokens, bias, K, V):
        return self.m.polish(tokens, bias, K, V)


def export(model: Echo, dtype=torch.float16) -> dict:
    """{"prefill", "step", "polish"} ExportedPrograms with dynamic NT, R, T and P."""
    import executorch.backends.mlx.custom_ops  # noqa: F401  registers mlx::rope

    v2 = EchoV2(model).eval().to(dtype)
    Dim = torch.export.Dim
    nt, r = Dim("NT", min=2, max=NT_MAX), Dim("R", min=2, max=R_MAX)
    t, p = Dim("T", min=2, max=T_MAX), Dim("P", min=4, max=P_MAX)
    cfg = model.cfg
    kv = torch.zeros(cfg["num_hidden_layers"], 1, cfg["num_key_value_heads"], 120, cfg["head_dim"], dtype=dtype)
    tokens = torch.full((model.n_cb, 100), model.mask_id)
    ex = torch.export.export
    return {
        "prefill": ex(_Prefill(v2), (torch.zeros(40, dtype=torch.long), torch.zeros(model.n_cb, 80, dtype=torch.long)),
                      dynamic_shapes=({0: nt}, {1: r}), strict=False),
        "step": ex(_Step(v2), (tokens, kv, kv.clone()), dynamic_shapes=({1: t}, {3: p}, {3: p}), strict=False),
        "polish": ex(_Polish(v2), (tokens, torch.zeros(1, 1, 1, 120, dtype=dtype), kv, kv.clone()),
                     dynamic_shapes=({1: t}, {3: p}, {3: p}, {3: p}), strict=False),
    }
