"""
Export wrappers for the LFM2.5 retrieval models:

  - LiquidAI/LFM2.5-Embedding-350M  (dense bi-encoder, CLS pooling, dim 1024)
  - LiquidAI/LFM2.5-ColBERT-350M    (late-interaction, Linear 1024->128 per token)

Both share the same Lfm2BidirectionalModel backbone (custom remote code: full
bidirectional attention + non-causal centered short-conv). We wrap the backbone
with the head so the exported graph emits the final, L2-normalized embedding(s),
matching the sentence-transformers / pylate reference post-processing.

The wrappers take (input_ids, attention_mask) of a fixed sequence length. The
caller is responsible for prepending the role prefix ("query: "/"document: " for
Embedding, "[Q] "/"[D] " for ColBERT) and for padding/masking. attention_mask is
the standard 2D padding mask; the backbone turns it into a bidirectional
additive mask internally.
"""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


EMBEDDING_MODEL_ID = "LiquidAI/LFM2.5-Embedding-350M"
COLBERT_MODEL_ID = "LiquidAI/LFM2.5-ColBERT-350M"


class Lfm2EmbeddingWrapper(nn.Module):
    """CLS pooling (index 0) + L2 normalize -> [B, 1024]."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(cls, p=2, dim=-1)
        # Always emit fp32: internal compute may be bf16 (MLX), but the runner /
        # JS side reads the output buffer as a Float32Array. A bf16 output would
        # be byte-reinterpreted as garbage floats.
        return emb.float()


class Lfm2ColbertWrapper(nn.Module):
    """Linear(1024->128, bias=False) per token + per-token L2 normalize -> [B, S, 128]."""

    def __init__(self, backbone: nn.Module, dense: nn.Linear):
        super().__init__()
        self.backbone = backbone
        self.dense = dense

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        tokens = self.dense(out.last_hidden_state)
        emb = torch.nn.functional.normalize(tokens, p=2, dim=-1)
        # Emit fp32 even when compute is bf16 (see Lfm2EmbeddingWrapper).
        return emb.float()


def load_embedding(dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(
        EMBEDDING_MODEL_ID, trust_remote_code=True, dtype=dtype
    ).eval()
    return Lfm2EmbeddingWrapper(backbone).eval(), tok


def load_colbert(dtype=torch.float32):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    tok = AutoTokenizer.from_pretrained(COLBERT_MODEL_ID, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(
        COLBERT_MODEL_ID, trust_remote_code=True, dtype=dtype
    ).eval()

    dense = nn.Linear(1024, 128, bias=False)
    sd = load_file(hf_hub_download(COLBERT_MODEL_ID, "1_Dense/model.safetensors"))
    dense.load_state_dict({"weight": sd["linear.weight"].to(dtype)})
    dense = dense.to(dtype).eval()

    return Lfm2ColbertWrapper(backbone, dense).eval(), tok


def patch_shortconv_static_length(model):
    """Replace the non-causal short-conv forward with one that drops the
    data-dependent `conv_out[..., :seq_len]` slice. For kernel size 3 with
    centered padding (pad=1) the conv output length always equals the input
    length, so the slice is a no-op -- but it produces a dynamic (`?`) shape
    that CoreML's runtime refuses to execute under flexible/RangeDim inputs
    (error 0x32). Removing it keeps numerics identical and makes the graph
    statically shaped, which CoreML can run dynamically (like the shipped
    MiniLM embedding models). No-op / unused for XNNPACK and MLX."""
    import torch.nn.functional as F

    def _forward(self, hidden_states, past_key_values=None, cache_position=None,
                 attention_mask=None):
        from transformers.models.lfm2.modeling_lfm2 import (
            apply_mask_to_padding_states,
        )

        x = apply_mask_to_padding_states(hidden_states, attention_mask)
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        Bx = B * x
        k = self.conv.weight.shape[-1]
        conv_out = F.conv1d(
            Bx, weight=self.conv.weight, bias=self.conv.bias,
            stride=1, padding=k // 2, dilation=1, groups=Bx.shape[1],
        )
        y = C * conv_out
        return self.out_proj(y.transpose(-1, -2).contiguous())

    for mod in model.modules():
        if mod.__class__.__name__ == "Lfm2ShortConv":
            mod.slow_forward = _forward.__get__(mod, mod.__class__)
            mod.forward = _forward.__get__(mod, mod.__class__)
    return model


# Role prefixes (mandatory; the models are trained with these).
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_DOC_PREFIX = "document: "
COLBERT_QUERY_PREFIX = "[Q] "
COLBERT_DOC_PREFIX = "[D] "


def encode_inputs(tok, texts, prefix, seq_len):
    """Prefix, tokenize, pad/truncate to a fixed seq_len. Returns (ids, mask)."""
    texts = [prefix + t for t in texts]
    enc = tok(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    return enc["input_ids"], enc["attention_mask"]
