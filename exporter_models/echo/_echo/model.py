"""Export-friendly Echo (software-mansion/Echo, pruned + distilled OmniVoice).

Echo is a bidirectional Qwen3 that fills masked audio-codec tokens. Its prefix
(style + text tokens + reference audio codes) never attends to the target, so
the prefix keys/values are computed once per clip and every sampler pass only
runs the target frames against them. That split needs no attention mask at all:
the prefix attends to itself, the target attends to everything.

Methods the exporters build from these modules (L = 24 layers, KV = 6 heads,
HD = 128, C = 8 codebooks, 1024 codes + 1 mask id):
  encode_prefix : (text i64[P], audio i64[C, P], is_audio f32[P], pos i64[P])
                  -> k_0, v_0, ..., k_{L-1}, v_{L-1}   each f32[KV, P, HD]
  denoise       : (codes i64[C, T], pos i64[T], bias f32[P], k_0, v_0, ...)
                  -> log_probs f32[C, T, 1024]
Every input dim is a single symbol (P or T): torch.export cannot declare a dim
as a sum of two others. So the prefix arrives as one sequence, text ids in
`text` and reference codes in `audio`, blended by `is_audio` (the slot that
does not apply holds any valid id, e.g. 0). Positions are passed in rather than
built with arange so the RoPE lookup is a plain gather on every backend; the
target continues at P. `bias` is added to every target score towards a prefix
key: zero on the main passes, +1 on the reference frames for the two polish
passes (target keys are never biased).

Fixed-size backends (QNN) pad P and T up to a bucket: `encode_prefix`'s
`key_bias` [P] and `denoise`'s `self_bias` [T] carry -1e4 on padded keys, and
`bias` carries it on padded prefix keys. Padded rows compute garbage the caller
ignores. -1e4 rather than -inf keeps fp16 softmax finite.

XNNPACK constraints shape the graph: keys/values travel as 2L separate tensors
because it cannot slice a dynamic dim (a stacked cache costs 2L CPU copies per
pass), every reshape names its dynamic dim as -1 (a symbolic size there is
rejected and splits the delegate), and RMSNorm's mean and RoPE's rotate_half
are constant linears (mean over the last dim and slices are not delegated).

Reference: `run/voice_cloning.ipynb` in the source repo (HF Qwen3Model + a
DynamicCache holding the prefix); the sampler lives in the caller.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Longest prefix + target the RoPE table covers (25 frames/s, so ~80 s).
MAX_POSITIONS = 2048


def dequantize(weights: dict, quantization: dict) -> dict:
    """A quantized checkpoint's codes -> floats (code * scale, per channel or per group).

    Verbatim semantics of the reference runner: int4 is packed two per byte, low
    nibble first, each stored as code + 8. Non-quantized tensors are bf16.
    """
    if quantization.get("rotation"):
        raise NotImplementedError("rotated checkpoints are not supported")
    group = quantization.get("group_size")
    for key in [k for k in weights if k.endswith("_scale")]:
        scale, codes = weights.pop(key), weights[key[: -len("_scale")]]
        if quantization["bits"] == 4:
            codes = torch.stack([codes & 0xF, codes >> 4], dim=-1).flatten(-2).float() - 8
        codes = codes.float()
        weights[key[: -len("_scale")]] = (
            (codes.unflatten(-1, (-1, group)) * scale[..., None]).flatten(-2) if group else codes * scale
        )
    return {k: v.float() if v.is_floating_point() else v for k, v in weights.items()}


class RMSNorm(nn.Module):
    # "native": aten.rms_norm, kept whole by QNN, MLX, Vulkan and Core ML, which
    # all guard the sum of squares (fp32 accumulation or max-abs prescaling).
    # Required on fp16 backends: the residual stream reaches |x| ~ 1.1e4, so
    # x * x (and even mean(x^2) ~ 1.8e5) overflows fp16.
    # "linear": mean of squares as a constant linear, for XNNPACK (fp32), which
    # neither takes rms_norm nor a mean over the last dim.
    # "vulkan": the decomposed x * rsqrt(mean(x * x, -1) + eps) spelling that
    # Vulkan's FusePatternsPass folds into its fp32-accumulating rms_norm kernel;
    # the Vulkan partitioner decomposes aten.rms_norm into pieces it then splits on.
    mode = "native"

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        # The mean of squares as a linear against a constant 1/dim column: XNNPACK
        # does not take mean over the last dim, and each one split the graph.
        self.register_buffer("mean_w", torch.full((1, dim), 1.0 / dim), persistent=False)
        # eps as a [1] tensor, not a Python scalar: coremltools' fuse_linear_bias
        # folds the add into the 1-output linear and indexes shape[-1] of a 0-d
        # bias (IndexError).
        self.register_buffer("eps_t", torch.tensor([eps]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if RMSNorm.mode == "native":
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        if RMSNorm.mode == "vulkan":
            return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.weight
        return x * torch.rsqrt(F.linear(x * x, self.mean_w) + self.eps_t) * self.weight


def _rotation(hd: int) -> torch.Tensor:
    """R with x @ R.T == rotate_half(x) = cat(-x[half:], x[:half]): a constant
    linear instead of two slices, which XNNPACK does not delegate."""
    half = hd // 2
    r = torch.zeros(hd, hd)
    r[torch.arange(half), torch.arange(half) + half] = -1.0
    r[torch.arange(half) + half, torch.arange(half)] = 1.0
    return r


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, hd = cfg["hidden_size"], cfg["head_dim"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = hd
        self.q_proj = nn.Linear(d, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, d, bias=False)
        self.q_norm = RMSNorm(hd, cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(hd, cfg["rms_norm_eps"])
        self.register_buffer("rot", _rotation(hd), persistent=False)

    rot_mode = "linear"  # "slice": cat(-x2, x1), for backends that slice natively (QNN)
    # QNN: concatenate keys rather than scores and expect the 1/sqrt(hd) scale already folded into
    # q_norm (fold_attention_scale). Same math; on the HTP the per-score mul/add/cat over
    # [kv, 2N, P+N] cost more than all the linears together.
    fused = False

    def rope(self, x, cos, sin):
        if Attention.rot_mode == "slice":
            half = self.hd // 2
            return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin
        return x * cos + F.linear(x, self.rot) * sin

    def kv(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Keys/values of h [N, D] as [KV, N, HD], keys already rotated."""
        k = self.k_norm(self.k_proj(h).view(-1, self.n_kv, self.hd)).transpose(0, 1)
        v = self.v_proj(h).view(-1, self.n_kv, self.hd).transpose(0, 1)
        return self.rope(k, cos, sin), v

    def forward(self, h, cos, sin, k, v, prefix=None, self_bias=None):
        """Queries from h [N, D] against its own keys/values [KV, N, HD] and, when
        given, cached prefix keys/values [KV, P, HD] plus an additive bias [P].
        `self_bias` [N] masks padded own keys in fixed-size (bucketed) exports."""
        q = self.q_norm(self.q_proj(h).view(-1, self.n_heads, self.hd)).transpose(0, 1)
        q = self.rope(q, cos, sin)  # [H, N, HD]
        # GQA without repeat_kv: query heads 2j and 2j+1 share kv head j, so fold
        # the pair into the row axis and run one bmm per kv head. Attention is
        # spelled bmm/softmax/bmm because SDPA's decomposition adds a masked-row
        # guard (eq/any/where) that no backend delegates.
        # .contiguous().view(-1) rather than reshape: export turns a reshape of a
        # strided tensor into a view with a symbolic size, which XNNPACK rejects.
        q = q.contiguous().view(self.n_kv, -1, self.hd)
        if Attention.fused:
            bias_all = self_bias
            if prefix is not None:
                kp, vp, bias = prefix
                k, v = torch.cat([kp, k], dim=1), torch.cat([vp, v], dim=1)
                own = self_bias if self_bias is not None else torch.zeros(q.shape[1] // (self.n_heads // self.n_kv), dtype=bias.dtype)
                bias_all = torch.cat([bias, own])
            scores = torch.bmm(q, k.transpose(1, 2))
            if bias_all is not None:
                scores = scores + bias_all
            out = torch.bmm(torch.softmax(scores, dim=-1), v)
            out = out.view(self.n_heads, -1, self.hd).transpose(0, 1).contiguous().view(-1, self.n_heads * self.hd)
            return self.o_proj(out)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.hd**-0.5
        if self_bias is not None:
            scores = scores + self_bias
        if prefix is not None:
            # Scores, not keys, are concatenated so the bias covers the prefix only.
            kp, vp, bias = prefix
            scores = torch.cat([torch.bmm(q, kp.transpose(1, 2)) * self.hd**-0.5 + bias, scores], dim=-1)
            v = torch.cat([vp, v], dim=1)
        out = torch.bmm(torch.softmax(scores, dim=-1), v)  # [KV, g*N, HD]
        out = out.view(self.n_heads, -1, self.hd).transpose(0, 1).contiguous().view(-1, self.n_heads * self.hd)
        return self.o_proj(out)


def fold_attention_scale(model):
    """For Attention.fused: scale q_norm's weight by 1/sqrt(hd) (RoPE is linear, so it commutes)."""
    with torch.no_grad():
        for layer in model.llm.layers:
            attn = layer.self_attn
            attn.q_norm.weight.mul_(attn.hd**-0.5)
    return model


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, f = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):  # weights only; the methods below drive it
    def __init__(self, cfg: dict):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.mlp = MLP(cfg)

class FactorizedEmbedding(nn.Module):
    """Echo's rank-256 text embedding: table lookup, then a 256 -> D projection."""

    def __init__(self, vocab: int, rank: int, dim: int):
        super().__init__()
        self.table = nn.Embedding(vocab, rank)
        self.project = nn.Linear(rank, dim, bias=False)

    def forward(self, ids):
        return self.project(self.table(ids))


class _LLM(nn.Module):
    def __init__(self, cfg: dict, rank: int):
        super().__init__()
        self.embed_tokens = FactorizedEmbedding(cfg["vocab_size"], rank, cfg["hidden_size"])
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])


class Echo(nn.Module):
    # "one": a single lookup with per-codebook offsets, then a sum over codebooks (QNN: 4.7x
    # faster than eight lookups); "per_codebook" for backends that cannot reduce over dim 0.
    embed_mode = "per_codebook"

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["llm_config"]
        self.cfg = cfg
        self.n_cb = config["num_audio_codebook"]
        self.vocab = config["audio_vocab_size"]
        self.mask_id = config["audio_mask_id"]
        d = cfg["hidden_size"]
        self.llm = _LLM(cfg, config["echo"]["architecture"]["embedding_rank"])
        self.audio_embeddings = nn.Embedding(self.n_cb * self.vocab, d)
        self.audio_heads = nn.Linear(d, self.n_cb * self.vocab, bias=False)
        self.register_buffer("codebook_layer_offsets", torch.arange(self.n_cb) * self.vocab)

        hd = cfg["head_dim"]
        theta = cfg["rope_parameters"]["rope_theta"]
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
        freqs = torch.arange(MAX_POSITIONS, dtype=torch.float64)[:, None] * inv[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("rope_cos", emb.cos().float(), persistent=False)
        self.register_buffer("rope_sin", emb.sin().float(), persistent=False)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "Echo":
        from safetensors.torch import load_file

        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        model = cls(config)
        sd = load_file(str(path / "model.safetensors"))
        if config.get("quantization"):
            sd = dequantize(sd, config["quantization"])
        # Centering each codebook's head rows leaves its softmax unchanged and keeps
        # the logits small enough for half precision (as the reference does).
        heads = sd["audio_heads.weight"].view(model.n_cb, model.vocab, -1)
        sd["audio_heads.weight"] = (heads - heads.mean(1, keepdim=True)).flatten(0, 1)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.endswith(("rope_cos", "rope_sin", "mean_w", "eps_t", "rot"))]
        if missing or unexpected:
            raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
        return model.eval()

    # -- building blocks --------------------------------------------------

    def rope(self, pos: torch.Tensor):
        # [N] -> [1, N, HD], broadcasting over heads.
        return F.embedding(pos, self.rope_cos).unsqueeze(0), F.embedding(pos, self.rope_sin).unsqueeze(0)

    def embed_audio(self, codes: torch.Tensor) -> torch.Tensor:
        # [C, N] -> [N, D]: one embedding row per codebook, summed. One lookup per
        # codebook plus adds, since neither XNNPACK nor Vulkan reduces over dim 0.
        if Echo.embed_mode == "one":
            return self.audio_embeddings(codes + self.codebook_layer_offsets[:, None]).sum(0)
        x = self.audio_embeddings(codes[0])
        for c in range(1, self.n_cb):
            x = x + self.audio_embeddings(codes[c] + c * self.vocab)
        return x

    def split_main_head(self, ncb=2):
        # Real-id and mask-id rows of the first ncb codebooks as their own constants, so the
        # exported graph holds no slice of the 1025-wide head.
        w = self.audio_heads.weight.detach().view(self.n_cb, self.vocab, -1)[:ncb]
        self.register_buffer("head_real", w[:, : self.mask_id].flatten(0, 1).contiguous(), persistent=False)
        self.register_buffer("head_mask", w[:, self.mask_id].contiguous(), persistent=False)

    def main_head(self, x, ncb=2):
        """What a main sampling pass reads, for codebooks 0..ncb-1: the argmax over the real ids
        and its log-prob under the full softmax (mask id included), as [ncb, N] int32 / float.

        The mask row is split off so every wide tensor is 1024 wide (HVX-aligned):
        max log p = ls_top - softplus(mask - lse_real), ls_top = top - lse_real."""
        if not hasattr(self, "head_real"):
            self.split_main_head(ncb)
        real = F.linear(x, self.head_real).view(-1, ncb, self.mask_id)
        mask = F.linear(x, self.head_mask)
        top = real.amax(-1)
        ls_top = torch.log_softmax(real, dim=-1).amax(-1)
        z = mask - top + ls_top
        conf = ls_top - (F.relu(z) + torch.log(1 + torch.exp(-z.abs())))
        return real.argmax(-1).to(torch.int32).t(), conf.t()

    # -- exported methods ---------------------------------------------------

    def encode_prefix(self, text, audio, is_audio, pos, key_bias=None, return_hidden=False):
        a = is_audio.unsqueeze(-1)
        x = self.llm.embed_tokens(text) * (1.0 - a) + self.embed_audio(audio) * a
        cos, sin = self.rope(pos)
        kv = []
        for layer in self.llm.layers:
            h = layer.input_layernorm(x)
            k, v = layer.self_attn.kv(h, cos, sin)
            x = x + layer.self_attn(h, cos, sin, k, v, self_bias=key_bias)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            kv += [k, v]
        # The last layer's output is never read; only its keys/values matter.
        # return_hidden keeps it as an extra output anyway, so the last layer's
        # attention survives dead-code elimination: in a quantized QNN build a
        # RoPE'd key that only feeds a graph output reads the gathered cos/sin
        # wrongly on the HTP (pytorch/executorch#24125).
        return (*kv, x) if return_hidden else tuple(kv)

    def denoise(self, codes, pos, bias, *kv, self_bias=None, head="logprobs"):
        x = self.embed_audio(codes)
        cos, sin = self.rope(pos)
        for i, layer in enumerate(self.llm.layers):
            h = layer.input_layernorm(x)
            k, v = layer.self_attn.kv(h, cos, sin)
            x = x + layer.self_attn(h, cos, sin, k, v, (kv[2 * i], kv[2 * i + 1], bias), self_bias)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self.llm.norm(x)
        if head == "raw":
            # [N, C * V] logits, token-major. Every read of the polish passes is an argmax that a
            # per-row constant does not move, so the log_softmax is not needed.
            return self.audio_heads(x)
        if head == "main":
            return self.main_head(x)
        logits = self.audio_heads(x).view(-1, self.n_cb, self.vocab).transpose(0, 1)
        # The mask id is never a prediction.
        return torch.log_softmax(logits, dim=-1)[..., : self.mask_id]
