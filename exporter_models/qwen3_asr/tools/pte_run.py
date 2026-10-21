"""Run a qwen3_asr .pte the way the RNE task does: preprocess -> per-window
audio_encoder -> token_embedding/text_decoder greedy loop. Writes {clip: text}.

usage: pte_run.py model.pte out.json clip.wav... (16 kHz mono int16)
Q3_STATIC=1 pads every window to 800 frames, for a static tower (Core ML).
"""
import json, math, os, sys, time, wave
import numpy as np, torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401  (custom sdpa / kv cache kernels)
import executorch.kernels.quantized  # noqa: F401
from transformers import AutoTokenizer

pte, out_json, clips = sys.argv[1], sys.argv[2], sys.argv[3:]
m = _load_for_executorch(pte)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B")
pre_ids = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>").input_ids
post_ids = tok("<|audio_end|><|im_end|>\n<|im_start|>assistant\n").input_ids
EOS = {151643, 151645}
STATIC = os.environ.get("Q3_STATIC") == "1"

def ntok(frames):
    rem = frames % 100
    tail = ((((rem - 1) // 2 + 1) - 1) // 2 + 1 - 1) // 2 + 1 if rem else 0
    return frames // 100 * 13 + tail

def run(name, *args):
    return m.run_method(name, tuple(a.contiguous() for a in args))[0]

def load(c):
    with wave.open(c) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
        return torch.from_numpy(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0)

res = {}
for c in clips:
    wav = load(c)
    t0 = time.time()
    mel = run("preprocess", wav)
    F = mel.shape[2]; pieces = []
    for w0 in range(0, F, 800):
        w = mel[:, :, w0:w0 + 800]; f = w.shape[2]
        w = torch.nn.functional.pad(w, (0, (800 if STATIC else math.ceil(f / 100) * 100) - f))
        n = ntok(f)
        pieces.append(run("audio_encoder", w, torch.tensor([n]))[0, :n])
    audio = torch.cat(pieces).unsqueeze(0)
    t1 = time.time()
    x = torch.cat([run("token_embedding", torch.tensor([pre_ids])), audio, run("token_embedding", torch.tensor([post_ids]))], dim=1)
    logits = run("text_decoder", x, torch.tensor([0]))
    pos = x.shape[1]; out = []; nxt = int(logits.argmax(-1))
    t2 = time.time()
    while nxt not in EOS and pos < 1024:
        out.append(nxt)
        logits = run("text_decoder", run("token_embedding", torch.tensor([[nxt]])), torch.tensor([pos]))
        pos += 1; nxt = int(logits.argmax(-1))
    t3 = time.time()
    raw = tok.decode(out)
    res[c.split("/")[-1][:-4]] = (raw.split("<asr_text>", 1)[1] if "<asr_text>" in raw else raw).strip()
    print(c.split("/")[-1], f"enc {t1-t0:.2f}s prefill {t2-t1:.2f}s ({x.shape[1]} tok) decode {1000*(t3-t2)/max(1,len(out)):.0f}ms/tok", repr(raw), flush=True)
json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False)
