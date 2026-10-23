"""Synthesize the source repo's test set with one backbone and score it with Whisper WER.

usage: python evaluate.py <ref:CKPT | eager:CKPT | pte:FILE> <out_dir> [--langs english polish] [--n 2]

  ref:CKPT    the reference model (transformers Qwen3Model + DynamicCache, as in
              run/voice_cloning.ipynb), CKPT = fp32 | w4a16 | w8a16 | w8a8
  eager:CKPT  the vendored export-friendly model (_echo/model.py)
  pte:FILE    an exported backbone .pte run through the ExecuTorch pybindings

The sampler (8 main passes + 2 polish passes), the target-length rule and the
silence trim follow the notebook. Audio is decoded with the eager codec and
transcribed by openai/whisper-small. Writes <out_dir>/<case>.wav and results.json.
"""

import argparse
import csv
import json
import math
import re
import sys
import tempfile
import time
import unicodedata
import wave
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _echo import hub  # noqa: E402

MASK = 1024
T_SHIFT = 0.1
CODEBOOK_PENALTY = 5.0
POSITION_TEMPERATURE = 5.0
REMASK_RATIO = 0.5
POLISH_BIAS = 1.0
POLISH_GUIDANCE = 0.5
WHISPER_LANG = {"en": "english", "de": "german", "fr": "french", "es": "spanish", "it": "italian",
                "pl": "polish", "pt": "portuguese"}


class ReferenceBackend:
    """The notebook's Echo: transformers Qwen3Model with a DynamicCache."""

    def __init__(self, checkpoint):
        from safetensors.torch import load_file
        from transformers import Qwen3Config, Qwen3Model

        from _echo.model import dequantize

        config = json.loads(Path(checkpoint, "config.json").read_text())
        weights = load_file(str(Path(checkpoint, "model.safetensors")))
        q = config.get("quantization")
        if q:
            weights = dequantize(weights, q)
        m = torch.nn.Module()
        m.llm = Qwen3Model(Qwen3Config.from_dict(config["llm_config"]))
        hidden = m.llm.config.hidden_size
        m.audio_embeddings = torch.nn.Embedding(8 * 1025, hidden)
        m.audio_heads = torch.nn.Linear(hidden, 8 * 1025, bias=False)
        if "llm.embed_tokens.table.weight" in weights:
            table = weights.pop("llm.embed_tokens.table.weight")
            weights["llm.embed_tokens.weight"] = table @ weights.pop("llm.embed_tokens.project.weight").T
        heads = weights["audio_heads.weight"].view(8, 1025, hidden)
        weights["audio_heads.weight"] = (heads - heads.mean(1, keepdim=True)).view(8 * 1025, hidden)
        weights.pop("codebook_layer_offsets", None)
        m.load_state_dict(weights)
        if q and q.get("activations"):
            def round_per_token(module, args):
                x = args[0]
                scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
                return ((x / scale).round().clamp(-127, 127) * scale,) + args[1:]

            for module in m.llm.layers.modules():
                if isinstance(module, torch.nn.Linear):
                    module.register_forward_pre_hook(round_per_token)
        self.m = m.eval()

    def embed_audio(self, codes):
        return self.m.audio_embeddings(codes + 1025 * torch.arange(8)[:, None]).sum(0, keepdim=True)

    def prefill(self, ids, ref):
        from transformers import DynamicCache

        prefix = torch.cat([self.m.llm.embed_tokens(ids[None]), self.embed_audio(ref)], dim=1)
        n = prefix.size(1)
        cache = DynamicCache()
        self.m.llm(inputs_embeds=prefix, attention_mask=torch.zeros(1, 1, n, n), past_key_values=cache)
        return cache

    def log_probs(self, cache, tokens, reference_frames, bias=0.0):
        prefix, frames = cache.get_seq_length(), tokens.size(1)
        mask = torch.zeros(1, 1, frames, prefix + frames)
        mask[..., prefix - reference_frames: prefix] = bias
        hidden = self.m.llm(
            inputs_embeds=self.embed_audio(tokens),
            attention_mask=mask,
            position_ids=torch.arange(prefix, prefix + frames)[None],
            past_key_values=cache,
        ).last_hidden_state
        cache.crop(prefix)
        logits = self.m.audio_heads(hidden)[0].view(frames, 8, -1).transpose(0, 1)
        return logits.log_softmax(-1)[..., :MASK]


class ExportBackend:
    """Anything exposing the two exported methods: the vendored eager model or a loaded .pte."""

    def __init__(self, encode_prefix, denoise):
        self.encode_prefix, self.denoise = encode_prefix, denoise

    @classmethod
    def eager(cls, checkpoint):
        from _echo.model import Echo

        m = Echo.from_pretrained(checkpoint)
        return cls(m.encode_prefix, m.denoise)

    @classmethod
    def pte(cls, path):
        from executorch.runtime import Runtime

        prog = Runtime.get().load_program(str(path))
        enc, den = prog.load_method("encode_prefix"), prog.load_method("denoise")
        c = lambda *a: [t.contiguous() for t in a]  # noqa: E731  (the pybindings ignore strides)
        return cls(lambda *a: enc.execute(c(*a)), lambda *a: den.execute(c(*a))[0])

    def prefill(self, ids, ref):
        nt, nr = ids.numel(), ref.size(1)
        text = torch.cat([ids, torch.zeros(nr, dtype=torch.long)])
        audio = torch.cat([torch.zeros(8, nt, dtype=torch.long), ref], dim=1)
        is_audio = torch.cat([torch.zeros(nt), torch.ones(nr)])
        return [t.clone() for t in self.encode_prefix(text, audio, is_audio, torch.arange(nt + nr))], nt + nr

    def log_probs(self, state, tokens, reference_frames, bias=0.0):
        kv, p = state
        b = torch.zeros(p)
        b[p - reference_frames:] = bias
        return self.denoise(tokens, torch.arange(p, p + tokens.size(1)), b, *kv)


def generate(backend, state, frames, reference_frames, steps=8):
    tokens = torch.full((8, frames), MASK)
    codebook = torch.arange(8)[:, None]
    t = torch.linspace(0, 1, steps + 1)
    t = (T_SHIFT * t / (1 - (1 - T_SHIFT) * t)).tolist()
    for step in range(steps):
        confidence, prediction = backend.log_probs(state, tokens, reference_frames).max(-1)
        masked = (tokens == MASK) & (codebook < 2)
        revealed = (tokens != MASK) & (codebook < 2)
        last = step == steps - 1
        k = int(masked.sum()) if last else math.ceil(2 * frames * (t[step + 1] - t[step]))
        gumbel = -torch.log(-torch.log(torch.rand_like(confidence) + 1e-10) + 1e-10)
        score = (confidence - CODEBOOK_PENALTY * codebook) / POSITION_TEMPERATURE + gumbel
        chosen = score.masked_fill(~masked, -math.inf).flatten().topk(k).indices
        tokens.view(-1)[chosen] = prediction.view(-1)[chosen]
        if not last:
            count = min(round(REMASK_RATIO * k), int(revealed.sum()))
            withdrawn = confidence.masked_fill(~revealed, math.inf).flatten().topk(count, largest=False).indices
            tokens.view(-1)[withdrawn] = MASK
    first = backend.log_probs(state, tokens, reference_frames, POLISH_BIAS)
    tokens[2:4] = first[2:4].argmax(-1)
    second = backend.log_probs(state, tokens, reference_frames, POLISH_BIAS)
    tokens[4:] = (second[4:] + POLISH_GUIDANCE * (second[4:] - first[4:])).argmax(-1)
    return tokens


def estimate_frames(text, reference_text, reference_frames):
    def weight(s):
        return sum({"Z": 0.2, "M": 0.0, "P": 0.5, "S": 0.5, "N": 3.5}.get(unicodedata.category(c)[0], 1.0) for c in s)

    frames = weight(text) / (weight(reference_text) / reference_frames)
    if frames < 50:
        frames = 50 * (frames / 50) ** (1 / 3)
    return max(1, int(frames))


def trim_silence(audio, sample_rate, keep=0.1):
    window = sample_rate // 100
    count = len(audio) // window
    level = 20 * np.log10(np.sqrt((audio[: count * window].reshape(count, window) ** 2).mean(1)) + 1e-10)
    loud = np.flatnonzero(level >= -50)
    if len(loud) == 0:
        return audio
    sounds = np.split(loud, np.flatnonzero(np.diff(loud) > 20) + 1)
    if len(sounds) > 1 and len(sounds[0]) <= 12:
        sounds = sounds[1:]
    if len(sounds) > 1 and len(sounds[-1]) <= 12:
        sounds = sounds[:-1]
    margin = round(keep * 100)
    return audio[max(sounds[0][0] - margin, 0) * window: (sounds[-1][-1] + 1 + margin) * window]


def test_cases(root: Path, codes_dir: Path, languages=None, per_language=None):
    """(language, reference codes, reference text, target text) from samples/test_set."""
    test = root / "samples" / "test_set"
    for z in test.glob("*/codecs.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(codes_dir / z.parent.name)
    rows_in = list(csv.DictReader(open(test / "test_in_processed.csv"), delimiter="|"))
    rows_out = list(csv.DictReader(open(test / "test_out.csv"), delimiter="|"))
    by_speaker = {r["speaker_id"]: r for r in rows_in}
    cases, seen = [], {}
    for r in rows_out:
        ref = by_speaker[r["speaker_id"]]
        lang_dir = ref["name"].split("/")[0]
        if languages and lang_dir not in languages:
            continue
        if per_language and seen.get(lang_dir, 0) >= per_language:
            continue
        seen[lang_dir] = seen.get(lang_dir, 0) + 1
        cases.append(dict(language=r["language"], codes=codes_dir / ref["name"], ref_text=ref["transcript"],
                          text=r["text"], speaker=r["speaker_id"]))
    return cases


def write_wav(path, x, sr):
    x = np.clip(np.asarray(x, dtype=np.float32), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def words(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]", " ", s.lower())).split()


def word_errors(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1], len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend")
    ap.add_argument("out")
    ap.add_argument("--langs", nargs="*", default=None, help="test_set language dirs (english, polish, ...)")
    ap.add_argument("--n", type=int, default=2, help="cases per language")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    from transformers import Qwen2Tokenizer, pipeline

    from _echo.codec import load_codec

    kind, arg = args.backend.split(":", 1)
    backend = {
        "ref": lambda c: ReferenceBackend(hub.checkpoint(c)),
        "eager": lambda c: ExportBackend.eager(hub.checkpoint(c)),
        "pte": ExportBackend.pte,
    }[kind](arg)
    root = Path(hub.source_files(["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                                  "samples/test_set/*.csv", "samples/test_set/*/codecs.zip"]))
    tok = Qwen2Tokenizer.from_pretrained(root)
    codec = load_codec(hub.codec())
    sr = codec.config.sample_rate
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    tmp = tempfile.TemporaryDirectory()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, errors, total = [], 0, 0
    for i, c in enumerate(test_cases(root, Path(tmp.name), args.langs, args.n)):
        ref = torch.from_numpy(np.load(c["codes"])).long()
        prompt = (f"<|denoise|><|lang_start|>{c['language']}<|lang_end|><|instruct_start|>None<|instruct_end|>"
                  f"<|text_start|>{c['ref_text']} {c['text']}<|text_end|>")
        ids = tok(prompt, return_tensors="pt").input_ids[0]
        frames = estimate_frames(c["text"], c["ref_text"], ref.size(1))
        t0 = time.time()
        state = backend.prefill(ids, ref)
        t1 = time.time()
        torch.manual_seed(args.seed)
        tokens = generate(backend, state, frames, ref.size(1), args.steps)
        t2 = time.time()
        audio = codec.decode(tokens[None]).audio_values[0, 0].numpy()
        audio = trim_silence(audio[: -int(0.2 * sr)], sr)
        name = f"{i:02d}_{c['speaker']}"
        write_wav(out / f"{name}.wav", audio, sr)
        a16 = torch.nn.functional.interpolate(torch.from_numpy(audio).view(1, 1, -1), size=int(len(audio) * 16000 / sr),
                                              mode="linear").view(-1).numpy()
        hyp = asr({"raw": a16, "sampling_rate": 16000},
                  generate_kwargs={"language": WHISPER_LANG[c["language"]], "task": "transcribe"})["text"].strip()
        e, n = word_errors(c["text"], hyp)
        errors, total = errors + e, total + n
        p = ids.numel() + ref.size(1)
        results.append(dict(name=name, lang=c["language"], P=p, T=frames, text=c["text"], hyp=hyp, wer=e / n,
                            prefill_s=t1 - t0, sample_s=t2 - t1, tokens=tokens.tolist()))
        print(f"{name} P={p} T={frames} WER={100 * e / n:5.1f}%  prefill {t1 - t0:.2f}s sample {t2 - t1:.2f}s | {hyp}",
              flush=True)
    print(f"TOTAL WER {100 * errors / total:.2f}% over {len(results)} clips")
    json.dump(dict(backend=args.backend, wer=errors / total, results=results), open(out / "results.json", "w"),
              ensure_ascii=False)


if __name__ == "__main__":
    main()
