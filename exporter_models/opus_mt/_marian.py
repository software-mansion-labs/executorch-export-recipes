"""MarianMT (Helsinki-NLP Opus-MT) as two export-friendly methods.

``encode(input_ids, attention_mask) -> (k_0, v_0, ..., k_L-1, v_L-1)`` runs the
encoder and then
projects its states through every decoder block's cross-attention key and
value Linears, so the decoder never recomputes them. Opus-MT's cross K/V
depend on the source only, and recomputing them per generated token costs
2 * S * d^2 MACs per block, about 10x the rest of a decode step at a 64-token
source.

``decode(input_ids, cache_position, attention_mask, k_0, v_0, ...) -> logits``
runs one decoder step. Self-attention keeps its K/V in per-block buffers of
``max_target_len`` rows, written at ``cache_position`` and read whole under a
causal mask, so stale rows from an earlier sentence are masked out and the
cache never needs a reset.

Each cross K/V tensor is ``[n_heads, src_len, head_dim]``, already head-major,
so attention is a rank-3 ``bmm`` with no transpose of the source side. They
travel as separate tensors rather than one stacked tensor because slicing a
dynamic-shaped input is a portable copy XNNPACK will not take.

``attention_mask`` (1 = token, 0 = padding) lets a backend that needs a
static source length pad the input; with a dynamic length it is all ones.

The architecture is HF ``MarianMTModel`` with the Opus-MT settings: post-LN
blocks, SiLU FFN, sinusoidal positions, sqrt(d)-scaled embeddings tied to the
output projection plus ``final_logits_bias``. The pad token (also the decoder
start token) is never a valid output (HF ``bad_words_ids``), so its logit is
pinned to a large negative in the bias.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# HF loads every Opus-MT checkpoint with this many positions.
MAX_POSITIONS = 512

# Additive mask value. Large enough to zero a softmax term next to logits of
# order 10, small enough to stay finite in fp16.
_MASK = -1.0e4


class _LayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` that can lower as primitives XNNPACK delegates.

    ExecuTorch's XNNPACK partitioner has no layer_norm, and its ``mean`` only
    reduces the two innermost dims of a 4D tensor, so with ``decomposed`` set
    the normalization runs over ``x.unsqueeze(-1)``. That keeps every encoder
    and decoder block inside one delegate instead of splitting it at each of
    its two or three norms. Backends with a native layer norm (Core ML, MLX,
    Vulkan) leave it unset.
    """

    decomposed = False

    def forward(self, x):
        if not self.decomposed:
            return super().forward(x)
        x4 = x.unsqueeze(-1)
        centered = x4 - x4.mean(dim=(-2, -1), keepdim=True)
        var = (centered * centered).mean(dim=(-2, -1), keepdim=True)
        normed = (centered * torch.rsqrt(var + self.eps)).reshape(1, -1, x.shape[-1])
        return normed * self.weight + self.bias


class _Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    # Reshapes spell the sequence length as -1: XNNPACK takes one inferred dim
    # but no symbolic ones, so a dynamic-length view would split the delegate.
    def heads(self, x: torch.Tensor) -> torch.Tensor:
        """``[1, T, d]`` -> ``[H, T, head_dim]``."""
        return x.view(-1, self.n_heads, self.head_dim).transpose(0, 1)

    def attend(self, q, k, v, bias) -> torch.Tensor:
        """Rank-3 attention over ``[H, T, head_dim]``; returns ``[1, Tq, d]``."""
        scores = torch.bmm(q * self.scale, k.transpose(1, 2)) + bias
        out = torch.bmm(torch.softmax(scores, dim=-1), v)
        return out.transpose(0, 1).contiguous().view(1, -1, self.n_heads * self.head_dim)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.self_attn = _Attention(d_model, n_heads)
        self.self_attn_layer_norm = _LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = _LayerNorm(d_model)

    def forward(self, x, bias):
        a = self.self_attn
        attended = a.attend(a.heads(a.q_proj(x)), a.heads(a.k_proj(x)), a.heads(a.v_proj(x)), bias)
        x = self.self_attn_layer_norm(x + a.out_proj(attended))
        x = self.final_layer_norm(x + self.fc2(F.silu(self.fc1(x))))
        return x


class _DecoderLayer(nn.Module):
    # "index_put" keeps the self-attention cache in plain buffers written with
    # index_put (portable on XNNPACK, model state on Core ML). "mlx" uses the
    # MLX delegate's mlx::kv_cache_update + mlx::custom_sdpa, which keep the
    # cache write and the attention inside the delegate. Set before building.
    cache_ops = "index_put"

    def __init__(self, d_model, n_heads, ffn_dim, max_target_len):
        super().__init__()
        self.self_attn = _Attention(d_model, n_heads)
        self.self_attn_layer_norm = _LayerNorm(d_model)
        self.encoder_attn = _Attention(d_model, n_heads)
        self.encoder_attn_layer_norm = _LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = _LayerNorm(d_model)
        head_dim = d_model // n_heads
        shape = (n_heads, max_target_len, head_dim)
        if self.cache_ops == "mlx":
            shape = (1,) + shape  # BHSD, as the MLX cache ops expect
        self.register_buffer("k_cache", torch.zeros(shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape), persistent=False)

    def self_attend(self, x, cache_position, causal_bias):
        a = self.self_attn
        q, k, v = a.heads(a.q_proj(x)), a.heads(a.k_proj(x)), a.heads(a.v_proj(x))
        if self.cache_ops == "mlx":
            from executorch.backends.mlx import custom_ops  # noqa: F401  (registers mlx::*)

            start_pos = cache_position[0].item()
            torch._check(start_pos >= 0)
            torch._check(start_pos < self.k_cache.shape[2])
            torch.ops.mlx.kv_cache_update(self.k_cache, k.unsqueeze(0), start_pos)
            torch.ops.mlx.kv_cache_update(self.v_cache, v.unsqueeze(0), start_pos)
            out = torch.ops.mlx.custom_sdpa(
                q.unsqueeze(0), self.k_cache, self.v_cache, start_pos=start_pos,
                is_causal=True, scale=a.scale,
            )
            return out[0].transpose(0, 1).contiguous().view(1, -1, a.n_heads * a.head_dim)
        # index_put with a leading null index: portable has no index_copy.
        self.k_cache[:, cache_position] = k
        self.v_cache[:, cache_position] = v
        return a.attend(q, self.k_cache, self.v_cache, causal_bias)

    def forward(self, x, cache_position, causal_bias, cross_k, cross_v, cross_bias):
        x = self.self_attn_layer_norm(x + self.self_attn.out_proj(self.self_attend(x, cache_position, causal_bias)))

        c = self.encoder_attn
        attended = c.attend(c.heads(c.q_proj(x)), cross_k, cross_v, cross_bias)
        x = self.encoder_attn_layer_norm(x + c.out_proj(attended))

        x = self.final_layer_norm(x + self.fc2(F.silu(self.fc1(x))))
        return x


class Marian(nn.Module):
    def __init__(self, cfg: dict, max_target_len: int):
        super().__init__()
        d = cfg["d_model"]
        self.d_model = d
        self.n_layers = cfg["decoder_layers"]
        self.n_heads = cfg["decoder_attention_heads"]
        self.pad_token_id = cfg["pad_token_id"]
        self.embed_scale = math.sqrt(d) if cfg.get("scale_embedding") else 1.0
        self.max_target_len = max_target_len
        # The dtype the graph computes in; the methods' I/O stays fp32.
        self.compute_dtype = torch.float32
        vocab = cfg["vocab_size"]

        self.embed_tokens = nn.Embedding(vocab, d)
        self.register_buffer("embed_positions", torch.zeros(MAX_POSITIONS, d))
        self.encoder_layers = nn.ModuleList(
            _EncoderLayer(d, cfg["encoder_attention_heads"], cfg["encoder_ffn_dim"])
            for _ in range(cfg["encoder_layers"])
        )
        self.decoder_layers = nn.ModuleList(
            _DecoderLayer(d, self.n_heads, cfg["decoder_ffn_dim"], max_target_len)
            for _ in range(self.n_layers)
        )
        # The tied output projection as its own Linear, so it can be quantized
        # without touching the input embedding lookup.
        self.lm_head = nn.Linear(d, vocab)
        causal = torch.full((max_target_len, max_target_len), _MASK).triu_(1)
        self.register_buffer("causal_bias", causal, persistent=False)

    def _padding_bias(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """``[1, S]`` int mask -> ``[1, 1, S]`` additive bias, broadcast over heads."""
        return ((1 - attention_mask.to(self.compute_dtype)) * _MASK).unsqueeze(0)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        src_len = input_ids.shape[1]
        x = self.embed_tokens(input_ids) * self.embed_scale + self.embed_positions[:src_len]
        bias = self._padding_bias(attention_mask)
        for layer in self.encoder_layers:
            x = layer(x, bias)

        # contiguous(): heads() is a transposed view, and torch.export records
        # an input or output with that layout as dim order (1, 0, 2). Callers
        # hand these tensors over as plain contiguous buffers.
        cross = []
        for layer in self.decoder_layers:
            c = layer.encoder_attn
            cross.append(c.heads(c.k_proj(x)).contiguous())
            cross.append(c.heads(c.v_proj(x)).contiguous())
        return tuple(cross)

    def decode(self, input_ids, cache_position, attention_mask, cross_kv) -> torch.Tensor:
        x = self.embed_tokens(input_ids) * self.embed_scale + self.embed_positions[cache_position]
        causal_bias = self.causal_bias[cache_position]
        cross_bias = self._padding_bias(attention_mask)
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, cache_position, causal_bias, cross_kv[2 * i], cross_kv[2 * i + 1], cross_bias)
        return self.lm_head(x)


class EncodeMethod(nn.Module):
    def __init__(self, model: Marian):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return tuple(t.float() for t in self.model.encode(input_ids, attention_mask))


class DecodeMethod(nn.Module):
    def __init__(self, model: Marian):
        super().__init__()
        self.model = model

    def forward(self, input_ids, cache_position, attention_mask, cross_kv):
        cross_kv = tuple(t.to(self.model.compute_dtype) for t in cross_kv)
        return self.model.decode(input_ids, cache_position, attention_mask, cross_kv).float()


def load(source_repo: str, max_target_len: int):
    """Build ``Marian`` from an HF Opus-MT checkpoint. Returns ``(model, hf_config)``."""
    from transformers import MarianMTModel

    hf = MarianMTModel.from_pretrained(source_repo, torch_dtype=torch.float32).eval()
    cfg = hf.config.to_dict()
    assert not cfg.get("normalize_before") and cfg["activation_function"] == "swish", cfg
    assert cfg["encoder_layers"] == cfg["decoder_layers"]

    model = Marian(cfg, max_target_len).eval().requires_grad_(False)
    enc, dec = hf.model.encoder, hf.model.decoder
    sd = {
        "embed_tokens.weight": hf.model.shared.weight,
        "embed_positions": enc.embed_positions.weight[:MAX_POSITIONS],
        "lm_head.weight": hf.lm_head.weight,
        "lm_head.bias": hf.final_logits_bias[0].clone(),
    }
    assert torch.equal(dec.embed_positions.weight, enc.embed_positions.weight)
    for prefix, layers in (("encoder_layers", enc.layers), ("decoder_layers", dec.layers)):
        for i, layer in enumerate(layers):
            for k, v in layer.state_dict().items():
                sd[f"{prefix}.{i}.{k}"] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.endswith(("k_cache", "v_cache", "causal_bias")) for k in missing), (
        missing, unexpected)
    with torch.no_grad():
        model.lm_head.bias[cfg["pad_token_id"]] = _MASK
    return model, cfg


def export_methods(model: Marian, cfg: dict, max_source_len: int, static: bool):
    """``torch.export`` both methods.

    ``static`` fixes the source length at ``max_source_len`` (the caller pads
    and masks); otherwise it is dynamic over ``1..max_source_len``.
    Returns ``({"encode": ep, "decode": ep}, schema_replacements)``.
    """
    from torch.export import Dim

    example = max_source_len if static else min(16, max_source_len)
    ids = torch.ones(1, example, dtype=torch.long)
    mask = torch.ones(1, example, dtype=torch.long)
    with torch.no_grad():
        cross_kv = EncodeMethod(model)(ids, mask)
    if static:
        enc_dyn = dec_dyn = None
    else:
        src = Dim("src_len", min=1, max=max_source_len)
        enc_dyn = {"input_ids": {1: src}, "attention_mask": {1: src}}
        dec_dyn = {"input_ids": None, "cache_position": None, "attention_mask": {1: src},
                   "cross_kv": tuple({1: src} for _ in cross_kv)}

    encode = torch.export.export(EncodeMethod(model), (ids, mask), dynamic_shapes=enc_dyn, strict=False)
    decode = torch.export.export(
        DecodeMethod(model),
        (torch.tensor([[cfg["decoder_start_token_id"]]]), torch.tensor([0]), mask, cross_kv),
        dynamic_shapes=dec_dyn,
        strict=False,
    )
    return {"encode": encode, "decode": decode}, schema(model, cfg, max_source_len, static)


def schema(model: Marian, cfg: dict, max_source_len: int, static: bool) -> dict:
    """The ``get_model_schema`` contract for these two methods."""

    def tensor(dtype, *dims):
        return {"kind": "Tensor", "dtype": dtype, "shape": [
            d if isinstance(d, dict) else {"kind": "constant", "value": d} for d in dims]}

    def same(*refs):
        return {"kind": "equality",
                "dims": [{"paramSide": s, "tensorIdx": i, "dimIdx": d} for s, i, d in refs]}

    src = {"kind": "range", "range": {"min": max_source_len if static else 1,
                                      "max": max_source_len, "step": 1}}
    n_kv = 2 * model.n_layers
    kv = tensor("float32", model.n_heads, src, model.d_model // model.n_heads)
    return {
        "encode": {
            "inputs": [tensor("int64", 1, src), tensor("int64", 1, src)],
            "outputs": [kv] * n_kv,
            "runtimeConstraints": [same(("input", 0, 1), ("input", 1, 1))]
            + [same(("input", 0, 1), ("output", i, 1)) for i in range(n_kv)],
        },
        "decode": {
            "inputs": [tensor("int64", 1, 1), tensor("int64", 1), tensor("int64", 1, src)] + [kv] * n_kv,
            "outputs": [tensor("float32", 1, 1, cfg["vocab_size"])],
            "runtimeConstraints": [same(("input", 2, 1), ("input", 3 + i, 1)) for i in range(n_kv)],
        },
    }


def schema_json(schema_dict: dict) -> str:
    """Validate against the repo's model.schema.json and serialize."""
    import json
    import tempfile

    from exporter.schema import load_and_validate_schema

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(schema_dict, f)
    return load_and_validate_schema(f.name)


def source_repo(size: str) -> str:
    """``en_de`` -> ``Helsinki-NLP/opus-mt-en-de``."""
    return f"Helsinki-NLP/opus-mt-{size.replace('_', '-')}"
