"""WER for a .pte whose methods need two different host runtimes.

The Core ML + MLX file runs its tower on Core ML and its decoder on MLX, and no
host Python build links both: the pip wheel has Core ML, a fork build has MLX.

stage enc: run preprocess + audio_encoder, save the audio embeddings.
stage dec: load them, run token_embedding + text_decoder greedily.

usage: split_run.py enc model.pte emb.pt clip.wav...   (Q3_STATIC=1 for a static tower)
       split_run.py dec model.pte emb.pt out.json
"""
import json
import math
import os
import sys
import wave

import numpy as np
import torch
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
from executorch.extension.pybindings.portable_lib import _load_for_executorch
import executorch.kernels.quantized  # noqa: F401

EOS = (151643, 151645)
STATIC = os.environ.get("Q3_STATIC") == "1"

stage, pte, emb_path = sys.argv[1:4]
m = _load_for_executorch(pte)


def run(name, *args):
    return m.run_method(name, tuple(a.contiguous() for a in args))[0]


def ntok(frames):
    rem = frames % 100
    tail = ((((rem - 1) // 2 + 1) - 1) // 2 + 1 - 1) // 2 + 1 if rem else 0
    return frames // 100 * 13 + tail


def encode(clip):
    with wave.open(clip) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    mel = run("preprocess", torch.from_numpy(pcm.astype(np.float32) / 32768.0))
    pieces = []
    for w0 in range(0, mel.shape[2], 800):
        win = mel[:, :, w0:w0 + 800]
        f = win.shape[2]
        width = 800 if STATIC else math.ceil(f / 100) * 100
        n = ntok(f)
        pieces.append(run("audio_encoder", torch.nn.functional.pad(win, (0, width - f)), torch.tensor([n]))[0, :n])
    return torch.cat(pieces).unsqueeze(0)


def decode(audio, pre, post):
    x = torch.cat([run("token_embedding", torch.tensor([pre])), audio, run("token_embedding", torch.tensor([post]))], dim=1)
    nxt = int(run("text_decoder", x, torch.tensor([0])).argmax(-1))
    pos, out = x.shape[1], []
    while nxt not in EOS and pos < 1024:
        out.append(nxt)
        nxt = int(run("text_decoder", run("token_embedding", torch.tensor([[nxt]])), torch.tensor([pos])).argmax(-1))
        pos += 1
    return out


if stage == "enc":
    torch.save({c.split("/")[-1][:-4]: encode(c) for c in sys.argv[4:]}, emb_path)
else:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B")
    pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>").input_ids
    post = tok("<|audio_end|><|im_end|>\n<|im_start|>assistant\n").input_ids
    res = {}
    for name, audio in torch.load(emb_path).items():
        raw = tok.decode(decode(audio, pre, post))
        res[name] = (raw.split("<asr_text>", 1)[1] if "<asr_text>" in raw else raw).strip()
        print(name, repr(raw), flush=True)
    json.dump(res, open(sys.argv[4], "w"), indent=1, ensure_ascii=False)
