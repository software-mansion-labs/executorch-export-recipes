"""Eager check of export_qnn_v3's prefill / step / polish (slot layout, in-graph positions and masks) against v3's
arithmetic on the proven v1 path: encode_prefix on a contiguous prefix, denoise's raw logits, then Step's / Polish's
math from _echo/v3.py in fp32. Random text, reference and target at L 100, R 200, 120 frames.

usage: python qnn/check_v3.py   (run from exporter_models/echo)
"""
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
from export_qnn_v3 import NEG, POLISH_BIAS, POLISH_GUIDANCE, PREFIX, build, polish_with_step  # noqa: E402

torch.manual_seed(0)
torch.set_grad_enabled(False)
model, methods = build()
L, R, F_, rf = 100, 200, 120, 200
i32 = torch.int32
mid, vocab = model.mask_id, model.vocab
text = torch.randint(0, 62035, (L,), dtype=i32)
ref = torch.randint(0, mid, (8, R), dtype=i32)

# ---- v3 wrappers (QNN layout: text slots [0, 512), reference slots [512, 1024))
prefill, ex = methods["prefill"]
text_in = torch.zeros(1024, dtype=i32); text_in[:L] = text
ref_in = torch.zeros(8, 1024, dtype=i32); ref_in[:, 512:512 + R] = ref
cache = torch.cat(prefill(text_in, ref_in, torch.tensor([L, R], dtype=i32)))


def target_tokens(T):
    tok = torch.full((8, T), mid, dtype=i32)
    tok[:2, : F_ // 2] = torch.randint(0, mid, (2, F_ // 2), dtype=i32)  # half of codebooks 0-1 filled, as mid-sampling
    tok[2:, :F_] = torch.randint(0, mid, (6, F_), dtype=i32)            # polish reads all codebooks filled
    return tok


lengths = torch.tensor([L, R, F_], dtype=i32)
main_lengths = torch.tensor([L, R, F_, 0], dtype=i32)  # a main pass: no reference frames biased towards

# ---- v1 reference: contiguous prefix [0, p), fp16-rounded KV like the cache
p = L + R
a_text = torch.zeros(PREFIX, dtype=torch.long); a_text[:L] = text.long()
a_audio = torch.zeros(8, PREFIX, dtype=torch.long); a_audio[:, L:p] = ref.long()
is_audio = torch.zeros(PREFIX); is_audio[L:p] = 1
key_bias = torch.full((PREFIX,), NEG); key_bias[:p] = 0
kv = [t.half().float() for t in model.encode_prefix(a_text, a_audio, is_audio, torch.arange(PREFIX), key_bias)]


def raw(tok, bias):
    T = tok.shape[1]
    self_bias = torch.full((T,), NEG); self_bias[:F_] = 0
    out = model.denoise(tok.long(), p + torch.arange(T), bias, *kv, self_bias=self_bias, head="raw")
    return out.view(T, 8, vocab).float()


# cache check: each layer's real keys equal v1's (text slots then reference slots)
k0 = cache[:6].float()
k_ref = kv[0]
cache_err = max((k0[:, :L] - k_ref[:, :L]).abs().max().item(), (k0[:, 512:512 + R] - k_ref[:, L:p]).abs().max().item())
print(f"prefill: layer-0 keys max |diff| to v1 {cache_err:.2e}")

for name in ("step_t128", "step_t256", "step"):
    mod, ex = methods[name]
    T = ex[0].shape[1]
    tok = target_tokens(T)
    prior = torch.zeros(T, 4, mid, dtype=torch.float16)            # read only by polish passes
    conf, pred, held = mod(tok, list(cache.split(6)), main_lengths, prior)[:3]
    logits = raw(tok, key_bias)[:, :2]                           # [T, 2, 1025]
    norm = torch.logsumexp(logits, -1)
    real = logits[..., :mid]
    r_conf, r_pred = (real.amax(-1) - norm).t(), real.argmax(-1).t()
    hold = tok[:2].t().long()
    r_held = (real.gather(-1, hold.clamp(max=mid - 1).unsqueeze(-1)).squeeze(-1) - norm).t()
    f = slice(0, F_)
    filled = (tok[:2] != mid)[:, f]
    print(f"{name}: prediction agreement {(pred[:, f] == r_pred[:, f]).float().mean():.4f}, "
          f"confidence max |diff| {(conf[:, f] - r_conf[:, f]).abs().max():.2e}, "
          f"held max |diff| (filled slots) {(held[:, f] - r_held[:, f])[filled].abs().max():.2e}")

# polish: what a caller does, two step calls (polish_with_step), at the smallest bucket
mod, ex = methods["step_t128"]
tok = target_tokens(ex[0].shape[1])
out = polish_with_step(mod, tok, torch.tensor([rf], dtype=i32), list(cache.split(6)), lengths)
pbias = key_bias.clone(); pbias[p - rf:p] += POLISH_BIAS
first = raw(tok, pbias)[:, 2:, :mid]
middle = first[:, :2].argmax(-1)
tok2 = torch.cat([tok[:2], middle.t().to(i32), tok[4:]])
second = raw(tok2, pbias)[:, 4:, :mid]
fine = ((1 + POLISH_GUIDANCE) * second - POLISH_GUIDANCE * first[:, 2:]).argmax(-1)
r_out = torch.cat([middle, fine], 1).t()
print(f"polish: token agreement {(out[:, :F_] == r_out[:, :F_].to(i32)).float().mean():.4f} over {6 * F_} tokens")
