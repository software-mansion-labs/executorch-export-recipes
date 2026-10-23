"""Calibration set for Echo's static QNN bucket: real prefixes and real sampler states.

usage: python calibrate.py <out.pt> [--checkpoint w8a16] [--P 320] [--T 256] [--n 12]

Each case pairs a dev-set reference (codes + transcript) with another dev
transcript of the same language as the target text, runs the reference sampler
(8 main passes + 2 polish passes, run/voice_cloning.ipynb in the source repo)
in eager, and keeps the encode_prefix inputs plus the denoise inputs of main
passes 0, 4, 7 and the first polish pass, padded to the bucket exactly as the
runtime pads them. Cases that do not fit the bucket are skipped.
"""

import argparse
import csv
import math
import random
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _echo import hub  # noqa: E402
from _echo.model import Echo  # noqa: E402

MASK, NEG = 1024, -1e4
i32 = torch.int32


def weight(s):
    return sum({"Z": 0.2, "M": 0.0, "P": 0.5, "S": 0.5, "N": 3.5}.get(unicodedata.category(c)[0], 1.0) for c in s)


def dev_set(dst: Path) -> tuple[list[dict], Path]:
    """samples/dev_set rows and the tokenizer path, with every language's
    codecs.zip unpacked under dst."""
    patterns = ["tokenizer.json", "samples/dev_set/dev_processed.csv", "samples/dev_set/*/codecs.zip"]
    root = Path(hub.source_files(patterns))
    dev = root / "samples" / "dev_set"
    for z in dev.glob("*/codecs.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(dst / z.parent.name)
    return list(csv.DictReader(open(dev / "dev_processed.csv"), delimiter="|")), root / "tokenizer.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--checkpoint", default="w8a16", help="pretrain/<name> in the source repo")
    ap.add_argument("--P", type=int, default=320)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--n", type=int, default=12, help="cases")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    from tokenizers import Tokenizer

    m = Echo.from_pretrained(hub.checkpoint(args.checkpoint))
    tmp = tempfile.TemporaryDirectory()
    codes_dir = Path(tmp.name)
    rows, tok_path = dev_set(codes_dir)
    tok = Tokenizer.from_file(str(tok_path))

    random.Random(0).shuffle(rows)
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["language"], []).append(r)

    samples, used = [], 0
    for r in rows:
        if used >= args.n:
            break
        others = [o for o in by_lang[r["language"]] if o is not r]
        text = random.Random(used).choice(others)["transcript"]
        ref = torch.from_numpy(np.load(codes_dir / r["name"])).long()
        prompt = (f"<|denoise|><|lang_start|>{r['language']}<|lang_end|><|instruct_start|>None<|instruct_end|>"
                  f"<|text_start|>{r['transcript']} {text}<|text_end|>")
        ids = torch.tensor(tok.encode(prompt, add_special_tokens=False).ids)
        nt, nr = ids.numel(), ref.size(1)
        p = nt + nr
        t = max(1, int(weight(text) / (weight(r["transcript"]) / nr)))
        if t < 50:
            t = int(50 * (t / 50) ** (1 / 3))
        if p > args.P or t > args.T:
            continue
        used += 1

        # The exact (unpadded) prefix drives the sampler; the padded copy is what gets saved.
        text_in = torch.cat([ids, torch.zeros(nr, dtype=torch.long)])
        audio_in = torch.cat([torch.zeros(8, nt, dtype=torch.long), ref], 1)
        is_audio = torch.cat([torch.zeros(nt), torch.ones(nr)])
        kv = m.encode_prefix(text_in, audio_in, is_audio, torch.arange(p))

        pad_p, pad_t = args.P - p, args.T - t
        key_bias = torch.cat([torch.zeros(p), torch.full((pad_p,), NEG)])
        pre = (torch.cat([text_in, torch.zeros(pad_p, dtype=torch.long)]).to(i32),
               torch.cat([audio_in, torch.zeros(8, pad_p, dtype=torch.long)], 1).to(i32),
               torch.cat([is_audio, torch.zeros(pad_p)]),
               torch.arange(args.P, dtype=i32), key_bias)
        kv_pad = m.encode_prefix(*(x.long() if x.dtype == i32 else x for x in pre[:4]), key_bias=pre[4])

        def den_inputs(tokens, bias_val):
            bias = torch.zeros(args.P)
            bias[p - nr: p] = bias_val
            bias[p:] = NEG
            self_bias = torch.cat([torch.zeros(t), torch.full((pad_t,), NEG)])
            codes = torch.cat([tokens, torch.full((8, pad_t), MASK)], 1).to(i32)
            pos = torch.cat([torch.arange(p, p + t), torch.zeros(pad_t, dtype=torch.long)]).to(i32)
            return (codes, pos, bias, self_bias, tuple(kv_pad))

        def log_probs(tokens, bias_val=0.0):
            b = torch.zeros(p)
            b[p - nr:] = bias_val
            return m.denoise(tokens, torch.arange(p, p + t), b, *kv)

        # Reference sampler, capturing selected passes.
        torch.manual_seed(used)
        tokens = torch.full((8, t), MASK)
        cb = torch.arange(8)[:, None]
        steps = 8
        ts = torch.linspace(0, 1, steps + 1)
        ts = (0.1 * ts / (1 - 0.9 * ts)).tolist()
        for step in range(steps):
            if step in (0, 4, 7):
                samples.append((pre, den_inputs(tokens.clone(), 0.0)))
            conf, pred = log_probs(tokens).max(-1)
            masked, revealed = (tokens == MASK) & (cb < 2), (tokens != MASK) & (cb < 2)
            last = step == steps - 1
            k = int(masked.sum()) if last else math.ceil(2 * t * (ts[step + 1] - ts[step]))
            g = -torch.log(-torch.log(torch.rand_like(conf) + 1e-10) + 1e-10)
            score = (conf - 5.0 * cb) / 5.0 + g
            ch = score.masked_fill(~masked, -math.inf).flatten().topk(k).indices
            tokens.view(-1)[ch] = pred.view(-1)[ch]
            if not last:
                cnt = min(round(0.5 * k), int(revealed.sum()))
                wd = conf.masked_fill(~revealed, math.inf).flatten().topk(cnt, largest=False).indices
                tokens.view(-1)[wd] = MASK
        samples.append((pre, den_inputs(tokens.clone(), 1.0)))
        print(f"case {used}: {r['language']} P={p} T={t}", flush=True)

    torch.save(samples, args.out)
    print(f"saved {len(samples)} samples from {used} cases to {args.out}")


if __name__ == "__main__":
    main()
