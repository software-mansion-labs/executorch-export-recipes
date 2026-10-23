"""Echo's two backbone methods as ExportedPrograms, shared by every backend.

  encode_prefix : (text i64[P], audio i64[8, P], is_audio f32[P], pos i64[P])
                  -> 48 x f32[6, P, 128]           (k_0, v_0, ..., k_23, v_23)
  denoise       : (codes i64[8, T], pos i64[T], bias f32[P], 48 x f32[6, P, 128])
                  -> log_probs f32[8, T, 1024]

P (prefix: text + reference frames) is dynamic in [8, 768] and T (target
frames) in [8, 1024]. I/O is fp32 whatever the compute dtype.
"""

import torch
from torch import nn

from .model import Echo, RMSNorm
from .quant import quantize_from_checkpoint

P_MAX = 768
T_MAX = 1024


class Prefix(nn.Module):
    def __init__(self, m, dtype):
        super().__init__()
        self.m, self.dtype = m, dtype

    def forward(self, text, audio, is_audio, pos):
        return tuple(t.float() for t in self.m.encode_prefix(text, audio, is_audio.to(self.dtype), pos))


class Denoise(nn.Module):
    def __init__(self, m, dtype):
        super().__init__()
        self.m, self.dtype = m, dtype

    def forward(self, codes, pos, bias, kv):
        return self.m.denoise(codes, pos, bias.to(self.dtype), *(t.to(self.dtype) for t in kv)).float()


def load(checkpoint: str, rms_norm: str, dtype=torch.float32, dynamic_quant: bool = False) -> Echo:
    """The checkpoint as an Echo, with `RMSNorm.mode` set for the backend and,
    for a quantized checkpoint, its own codes injected (int8 dynamic
    activations when dynamic_quant, the "8da" kernels)."""
    RMSNorm.mode = rms_norm
    model = Echo.from_pretrained(checkpoint)
    if dynamic_quant:
        quantize_from_checkpoint(model, checkpoint, dynamic=True)
    return model.to(dtype)


def export(model: Echo, dtype=torch.float32) -> dict:
    """{"encode_prefix": ep, "denoise": ep} with dynamic P and T."""
    P, T = 274, 120
    n_kv = 2 * len(model.llm.layers)
    kv_heads, head_dim = model.cfg["num_key_value_heads"], model.cfg["head_dim"]
    pre_in = (torch.zeros(P, dtype=torch.long), torch.zeros(8, P, dtype=torch.long), torch.zeros(P), torch.arange(P))
    den_in = (
        torch.full((8, T), 1024, dtype=torch.long),
        torch.arange(P, P + T),
        torch.zeros(P),
        tuple(torch.zeros(kv_heads, P, head_dim) for _ in range(n_kv)),
    )
    p = torch.export.Dim("P", min=8, max=P_MAX)
    t = torch.export.Dim("T", min=8, max=T_MAX)
    return {
        "encode_prefix": torch.export.export(
            Prefix(model, dtype).eval(), pre_in, dynamic_shapes=({0: p}, {1: p}, {0: p}, {0: p}), strict=False
        ),
        "denoise": torch.export.export(
            Denoise(model, dtype).eval(),
            den_in,
            dynamic_shapes=({1: t}, {0: t}, {0: p}, tuple({1: p} for _ in range(n_kv))),
            strict=False,
        ),
    }
