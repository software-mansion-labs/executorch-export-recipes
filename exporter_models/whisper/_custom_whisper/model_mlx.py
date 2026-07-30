# MIT License
#
# Copyright (c) 2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""MLX-delegate-friendly variant of the Whisper model.

This is a drop-in replacement for ``custom_whisper.model.TextDecoder`` whose
self-attention uses the MLX delegate's KV-cache primitives
(``mlx::kv_cache_update`` + ``mlx::custom_sdpa``) so that the KV-cache writes
and the attention computation are fully delegated to the MLX backend instead
of falling back to ExecuTorch's portable (CPU) runtime.

The forward signature ``forward(tokens, positions, encodings)`` and the I/O
shapes are identical to the original ``TextDecoder``, so the resulting
``.pte`` stays compatible with ``multi_test.py``.

Everything that does not change (encoder, preprocessor, LayerNorm, Linear,
sinusoids, dims) is reused from ``custom_whisper.model``.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Importing this module registers the mlx::kv_cache_update / mlx::custom_sdpa
# custom ops with torch's dispatcher, which is required for torch.export.
from executorch.backends.mlx import custom_ops as _mlx_custom_ops  # noqa: F401
from executorch.backends.mlx.llm.cache import KVCache

from .model import AudioEncoder, LayerNorm, Linear, ModelDimensions, Whisper, sinusoids

MAX_SEQ_LEN = 128


class MultiHeadAttentionMlx(nn.Module):
    """Self/cross attention whose KV cache and SDPA delegate to MLX.

    Self-attention: K/V are written into a BHSD ``KVCache`` via
    ``mlx::kv_cache_update`` and attention is computed with
    ``mlx::custom_sdpa`` (which slices the full cache to ``[0:stop_pos]`` and
    applies a causal mask internally).
    Cross-attention: K/V are recomputed from the encoder output each call and
    plain ``F.scaled_dot_product_attention`` (no causal mask) is used; this
    also delegates to MLX.
    """

    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool,
        max_cache_len: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_state // n_head
        self.scale = self.head_dim**-0.5
        self.cross_attention = cross_attention

        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        if not cross_attention:
            self.kv_cache = KVCache(
                max_batch_size=1,
                max_context_length=max_cache_len,
                n_heads=n_head,
                head_dim=self.head_dim,
                enable_dynamic_shape=True,
                dtype=dtype,
            )

    def forward(
        self,
        x: Tensor,
        start_pos: int,
        xa: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, _ = x.shape
        H, D = self.n_head, self.head_dim

        q = self.query(x)
        q = q.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

        if xa is None:
            k = self.key(x)
            v = self.value(x)
            k = k.view(B, T, H, D).transpose(1, 2)
            v = v.view(B, T, H, D).transpose(1, 2)

            k, v = self.kv_cache.update(start_pos, k, v)
            attn_out = torch.ops.mlx.custom_sdpa(
                q, k, v, start_pos=start_pos, is_causal=True, scale=self.scale
            )
        else:
            k = self.key(xa)
            v = self.value(xa)
            k = k.view(xa.shape[0], xa.shape[1], H, D).transpose(1, 2)
            v = v.view(xa.shape[0], xa.shape[1], H, D).transpose(1, 2)

            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=False, scale=self.scale
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out(attn_out)


class ResidualAttentionBlockMlx(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool,
        max_cache_len: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.attn = MultiHeadAttentionMlx(
            n_state, n_head, cross_attention=False, max_cache_len=max_cache_len, dtype=dtype
        )
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttentionMlx(
                n_state, n_head, cross_attention=True, max_cache_len=max_cache_len, dtype=dtype
            )
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        start_pos: int,
        xa: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_ln(x), start_pos)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), start_pos, xa)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class TextDecoderMlx(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        max_cache_len: int,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlockMlx(
                    n_state, n_head, cross_attention=True, max_cache_len=max_cache_len, dtype=dtype
                )
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)
        self.max_cache_len = max_cache_len

    @classmethod
    def from_decoder(
        cls, decoder: nn.Module, max_cache_len: int, dtype: torch.dtype
    ) -> "TextDecoderMlx":
        n_vocab, n_state = decoder.token_embedding.weight.shape
        n_ctx = decoder.positional_embedding.shape[0]
        n_head = decoder.blocks[0].attn.n_head
        n_layer = len(decoder.blocks)

        new = cls(n_vocab, n_ctx, n_state, n_head, n_layer, max_cache_len, dtype)
        new.token_embedding.weight = decoder.token_embedding.weight
        new.positional_embedding = decoder.positional_embedding
        for new_block, old_block in zip(new.blocks, decoder.blocks):
            new_block.attn.query.weight = old_block.attn.query.weight
            new_block.attn.query.bias = old_block.attn.query.bias
            new_block.attn.key.weight = old_block.attn.key.weight
            new_block.attn.value.weight = old_block.attn.value.weight
            new_block.attn.value.bias = old_block.attn.value.bias
            new_block.attn.out.weight = old_block.attn.out.weight
            new_block.attn.out.bias = old_block.attn.out.bias
            new_block.attn_ln.weight = old_block.attn_ln.weight
            new_block.attn_ln.bias = old_block.attn_ln.bias
            new_block.cross_attn.query.weight = old_block.cross_attn.query.weight
            new_block.cross_attn.query.bias = old_block.cross_attn.query.bias
            new_block.cross_attn.key.weight = old_block.cross_attn.key.weight
            new_block.cross_attn.value.weight = old_block.cross_attn.value.weight
            new_block.cross_attn.value.bias = old_block.cross_attn.value.bias
            new_block.cross_attn.out.weight = old_block.cross_attn.out.weight
            new_block.cross_attn.out.bias = old_block.cross_attn.out.bias
            new_block.cross_attn_ln.weight = old_block.cross_attn_ln.weight
            new_block.cross_attn_ln.bias = old_block.cross_attn_ln.bias
            new_block.mlp[0].weight = old_block.mlp[0].weight
            new_block.mlp[0].bias = old_block.mlp[0].bias
            new_block.mlp[2].weight = old_block.mlp[2].weight
            new_block.mlp[2].bias = old_block.mlp[2].bias
            new_block.mlp_ln.weight = old_block.mlp_ln.weight
            new_block.mlp_ln.bias = old_block.mlp_ln.bias
        new.ln.weight = decoder.ln.weight
        new.ln.bias = decoder.ln.bias
        return new

    def forward(
        self, tokens: torch.LongTensor, positions: torch.LongTensor, encodings: Tensor
    ) -> Tensor:
        B, T = tokens.shape
        start_pos = positions[0].item()
        torch._check(start_pos >= 0)
        torch._check(start_pos + T <= self.max_cache_len)

        x = self.token_embedding(tokens) + self.positional_embedding[positions]
        x = x.to(encodings.dtype)

        for block in self.blocks:
            x = block(x, start_pos, xa=encodings)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        return logits
