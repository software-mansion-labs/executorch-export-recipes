from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
import math
import torch


class AudioPreprocessor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.audio_preprocess = nn.Sequential(
            nn.Conv1d(1, dim, 127, 64, bias=False),
            nn.Tanh(),
            nn.GroupNorm(1, dim),
            nn.Conv1d(dim, 2 * dim, 7, 3),
            nn.GELU(),
            nn.Conv1d(2 * dim, dim, 3, 2),
            nn.GELU(),
            Rearrange("... c s -> ... s c"),
        )

    def forward(self, src):
        src = src.unsqueeze(-2)
        return self.audio_preprocess(src)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t):
        freqs = torch.einsum("i , j -> i j", t.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1)
        return rearrange(freqs, "... d r -> ... (d r)")


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(t, freqs):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[-seq_len:, :]

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = t * freqs.cos() + rotate_half(t) * freqs.sin()
    out = torch.cat((t, t_unrotated), dim=-1)

    return out.type(orig_dtype)

class FFLinearGelu(nn.Module):
    def __init__(self, dim, ff_mult=4):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult, bias=True),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim, bias=True),
        )

    def forward(self, x):
        return self.ff(x)


class FFSwiGLU(nn.Module):
    def __init__(self, dim, ff_mult=4):
        super().__init__()

        self.ff_proj = nn.Linear(dim, dim * ff_mult * 2, bias=True)
        self.ff_act = nn.SiLU()
        self.ff_out = nn.Linear(dim * ff_mult, dim, bias=True)

    def forward(self, x):
        x, gate = self.ff_proj(x).chunk(2, dim=-1)
        gate = gate * torch.sigmoid(gate)
        x = torch.mul(x, gate)
        return self.ff_out(x)


class KVCache(nn.Module):
    def __init__(self):
        super().__init__()
        max_seq_length = 30*6
        k_cache_shape = (6, 8, max_seq_length, 36) # (layers, heads, max_seq_length, dim//heads)
        v_cache_shape = (6, 8, max_seq_length, 36) # (layers, heads, max_seq_length, dim//heads)
        self.register_buffer("k_cache", torch.zeros(k_cache_shape))
        self.register_buffer("v_cache", torch.zeros(v_cache_shape))
        self.seq_pos = 0
        self.layer_pos = 0

    def reset_cache(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, inner_dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    # Scaled dot product attention
    def sdp_attention(self, q, k_t, v, mask=None):
        d_tensor = v.shape[3]

        op = (q @ k_t) / math.sqrt(d_tensor)
        if mask is not None:
            op = op.masked_fill(mask, -torch.finfo(op.dtype).max)

        score = self.softmax(op)
        out = score @ v

        # concat and pass to linear layer
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def forward(self, q, k, v, rot_pos_emb=None, mask=None):
        # dot product with weight matrices
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_head)

        # apply RoPE
        if rot_pos_emb is not None:
            q = apply_rotary_pos_emb(q, rot_pos_emb)
            k = apply_rotary_pos_emb(k, rot_pos_emb)
        k_t = k.transpose(2, 3)

        return self.sdp_attention(q, k_t, v, mask), k_t, v
    
    def forward_cached(self, q, k, v, kv_cache, layer_idx, seq_idx, rot_pos_emb, mask):
        # dot product with weight matrices
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_head)

        # apply RoPE
        q = apply_rotary_pos_emb(q, rot_pos_emb)
        k = apply_rotary_pos_emb(k, rot_pos_emb)

        kv_cache.k_cache[layer_idx:layer_idx+1,:,seq_idx:seq_idx+1,:] = k
        kv_cache.v_cache[layer_idx:layer_idx+1,:,seq_idx:seq_idx+1,:] = v
        k_t = kv_cache.k_cache[layer_idx,:,:seq_idx+1,:].unsqueeze(0).transpose(2,3)
        v = kv_cache.v_cache[layer_idx,:,:seq_idx+1,:].unsqueeze(0)
        return self.sdp_attention(q, k_t, v, mask=mask)



class MultiHeadCausalSelfAttentionWithKVCache(MultiHeadAttention):
    def __init__(self, dim, inner_dim, n_head):
        super().__init__(dim, inner_dim, n_head)

    def forward(self, q, k, v, kv_cache, layer_idx, seq_idx, rot_pos_emb, mask):
        # dot product with weight matrices
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_head)

        # apply RoPE
        q = apply_rotary_pos_emb(q, rot_pos_emb)
        k = apply_rotary_pos_emb(k, rot_pos_emb)

        kv_cache.k_cache[layer_idx:layer_idx+1,:,seq_idx:seq_idx+1,:] = k
        kv_cache.v_cache[layer_idx:layer_idx+1,:,seq_idx:seq_idx+1,:] = v
        k_t = kv_cache.k_cache[layer_idx,:,:seq_idx+1,:].unsqueeze(0).transpose(2,3)
        v = kv_cache.v_cache[layer_idx,:,:seq_idx+1,:].unsqueeze(0)
        return super().sdp_attention(q, k_t, v, mask=mask)


class MultiHeadCrossAttentionWithKVCache(MultiHeadAttention):
    def __init__(self, dim, inner_dim, n_head):
        super().__init__(dim, inner_dim, n_head)

    def forward(self, q, k_cache, v_cache):
        q = self.to_q(q)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_head)

        return super().sdp_attention(q, k_cache, v_cache)

class EncoderLayer(nn.Module):
    def __init__(self, dim, inner_dim, n_head, ff_swiglu, ff_mult=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, bias=False)

        self.attention = MultiHeadAttention(dim, inner_dim=inner_dim, n_head=n_head)

        self.norm2 = nn.LayerNorm(dim, bias=False)

        self.ff = FFSwiGLU(dim, ff_mult) if ff_swiglu else FFLinearGelu(dim, ff_mult)

    def forward(self, x, rot_pos_emb):
        _x = x
        x = self.norm1(x)
        x, _, _ = self.attention(q=x, k=x, v=x, rot_pos_emb=rot_pos_emb)
        x = x + _x

        _x = x
        x = self.norm2(x)
        x = self.ff(x)

        x = x + _x
        return x


class Encoder(nn.Module):
    def __init__(self, dim, inner_dim, n_head, n_layers, ff_swiglu):
        super().__init__()
        rot_embed_dim = max(inner_dim / n_head / 2, 32)
        self.rot_pos_emb = RotaryEmbedding(rot_embed_dim)

        self.layers = nn.ModuleList(
            [EncoderLayer(dim, inner_dim, n_head, ff_swiglu) for _ in range(n_layers)]
        )
        self.post_norm = nn.LayerNorm(dim, bias=False)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        rot_pos_emb = self.rot_pos_emb(pos)

        for layer in self.layers:
            x = layer(x, rot_pos_emb=rot_pos_emb)
        return self.post_norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, dim, inner_dim, n_head, ff_swiglu, ff_mult=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, bias=False)

        self.self_attention = MultiHeadAttention(
            dim, inner_dim=inner_dim, n_head=n_head
        )

        self.norm2 = nn.LayerNorm(dim, bias=False)
        self.cross_attention = MultiHeadAttention(
            dim, inner_dim=inner_dim, n_head=n_head
        )

        self.norm3 = nn.LayerNorm(dim, bias=False)
        self.ff = FFSwiGLU(dim, ff_mult) if ff_swiglu else FFLinearGelu(dim, ff_mult)

    def forward(self, x, context, rot_pos_emb):
        dim = x.size()[1]
        causal_mask = torch.ones((dim, dim), dtype=torch.bool).triu(1).to(x.device)
        _x = x
        x = self.norm1(x)
        x, new_k_cache, new_v_cache = self.self_attention(
            q=x,
            k=x,
            v=x,
            rot_pos_emb=rot_pos_emb,
            mask=causal_mask,
        )
        x = x + _x

        _x = x
        x = self.norm2(x)
        x, x_attn_k_cache, x_attn_v_cache = self.cross_attention(
            q=x, k=context, v=context
        )
        x = x + _x

        _x = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + _x

        return x, new_k_cache, new_v_cache, x_attn_k_cache, x_attn_v_cache

    def forward_cached(self, x, kv_cache, layer_idx, seq_idx, context, rot_pos_emb):
        dim = x.size()[1]
        causal_mask = torch.ones((dim, dim), dtype=torch.bool).triu(1).to(x.device)
        _x = x
        x = self.norm1(x)
        x = self.self_attention.forward_cached(
            q=x,
            k=x,
            v=x,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            seq_idx=seq_idx,
            rot_pos_emb=rot_pos_emb,
            mask=causal_mask,
        )
        x = x + _x

        _x = x
        x = self.norm2(x)
        x, _, _ = self.cross_attention(
            q=x, k=context, v=context
        )
        x = x + _x

        _x = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + _x

        return x


class Decoder(nn.Module):
    def __init__(self, dim, inner_dim, n_head, n_layers, dec_voc_size, ff_swiglu):
        super().__init__()

        self.n_head = n_head
        self.d_head = inner_dim // n_head

        rot_embed_dim = max(inner_dim / n_head / 2, 32)
        self.rot_pos_emb = RotaryEmbedding(rot_embed_dim)

        self.layers = nn.ModuleList(
            [DecoderLayer(dim, inner_dim, n_head, ff_swiglu) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(dim, bias=False)
        self.token_embedding = nn.Embedding(dec_voc_size, dim)
        self.kv_cache = KVCache()

    # def init_kv_cache(self)

    def forward_cached(self, x, enc_src):
        pos = torch.arange(x.shape[1], device=x.device)
        rot_pos_emb = self.rot_pos_emb(pos)
        x = self.token_embedding(x)

        seq_len = x.size(1)
        for idx, layer in enumerate(self.layers):
            x = layer.forward_cached(
                x[:, -1:],
                kv_cache=self.kv_cache,
                layer_idx=idx,
                seq_idx=seq_len,
                context=enc_src,
                rot_pos_emb=rot_pos_emb,
            )

        x = self.final_norm(x)

        return x @ self.token_embedding.weight.t()

    def forward(self, x, enc_src):
        pos = torch.arange(x.shape[1], device=x.device)
        rot_pos_emb = self.rot_pos_emb(pos)
        x = self.token_embedding(x)

        k_cache = []
        v_cache = []
        x_attn_k_cache = []
        x_attn_v_cache = []

        for idx, layer in enumerate(self.layers):
            x, new_k_line, new_v_line, new_x_attn_k_line, new_x_attn_v_line = layer(
                x[:,-1:],
                enc_src,
                rot_pos_emb,
            )

            k_cache.append(new_k_line)
            v_cache.append(new_v_line)
            # x_attn_k_cache.append(new_x_attn_k_line)
            # x_attn_v_cache.append(new_x_attn_v_line)

        x = self.final_norm(x)

        return (
            x @ self.token_embedding.weight.t(),
            *k_cache,
            *v_cache,
            # *x_attn_k_cache,
            # *x_attn_v_cache,
        )


class MoonshineModel(nn.Module):
    def __init__(
        self,
        dim,
        inner_dim,
        n_head,
        enc_depth,
        dec_depth,
        dec_voc_size=32768,
        enc_ff_swiglu=False,
        dec_ff_swiglu=True,
    ):
        super().__init__()
        self.preprocessor = AudioPreprocessor(dim)
        self.encoder = Encoder(
            dim, inner_dim, n_head, enc_depth, ff_swiglu=enc_ff_swiglu
        )

        self.decoder = Decoder(
            dim, inner_dim, n_head, dec_depth, dec_voc_size, ff_swiglu=dec_ff_swiglu
        )
        self.dec_depth = dec_depth
        self.n_head = n_head
        self.d_head = inner_dim // n_head

    def generate(self, src, seq_len):
        preprocessed = self.preprocessor(src)
        enc = self.encoder(preprocessed)
        sot_token = 1
        eot_token = 2

        seq = torch.as_tensor([[sot_token]]).to(src.device)

        vals = self.decoder(x=seq, enc_src=enc)
        logits = vals[0]
        k_cache = vals[1 : 1+self.dec_depth]
        v_cache = vals[1+self.dec_depth : 1+2*self.dec_depth]

        self.decoder.kv_cache.k_cache[:,:,:1,:] = torch.concat(k_cache, dim=0).transpose(2,3)
        self.decoder.kv_cache.v_cache[:,:,:1,:] = torch.concat(v_cache, dim=0)

        sample = logits[:, -1].argmax(dim=-1, keepdim=True)
        seq = torch.cat((seq, sample), dim=-1)

        while sample != eot_token and len(seq.flatten()) <= seq_len:
            vals = self.decoder.forward_cached(
                seq,
                enc,
            )
            logits = vals
            logits = logits[:, -1]  # get last token
            sample = logits.argmax(dim=-1, keepdim=True)
            seq = torch.cat((seq, sample), dim=-1)

        return seq
