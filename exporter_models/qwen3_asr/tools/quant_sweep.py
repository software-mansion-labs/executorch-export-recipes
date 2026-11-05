"""Decoder quantization sweep: first-token log p("language") per recipe.

The model opens every transcript with `language <Name><asr_text>`; the fp32
decoder puts that first token at log p ~ 0 on every clip. A recipe that drops
it is one that derails transcripts (see NOTES.md), so this is a cheap proxy for
WER that needs no export.

A recipe is ';'-separated `regex=scheme` rules matched against decoder linear
names (first match wins; `default` matches everything). Schemes: `none`,
`8da4w-g<N>`, `8da8w-pc` (XNNPACK/Vulkan style), `4w-g<N>`, `8w-pc` (weight-only,
MLX style). Set DTYPE=bf16 to run the decoder in bf16.

    uv run python tools/quant_sweep.py a.wav,b.wav \\
        "default=none" "output=8da8w-pc;layers\\.(0|1)\\.=8da8w-pc;default=8da4w-g32"
"""

import importlib.util
import math
import os
import re
import sys
import wave
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

import _qwen3_asr as q  # noqa: E402
import _qwen3_asr.exportable  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location("qwen3_asr_xnnpack_export", _HERE / "xnnpack" / "export.py")
xe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xe)


def _config(scheme: str):
    if scheme.startswith("8da"):
        return xe._linear_config(scheme)
    from torchao.quantization import MappingType
    from torchao.quantization.granularity import PerAxis, PerGroup
    from torchao.quantization.quant_api import IntxWeightOnlyConfig

    bits, _, gran = scheme.partition("-")
    return IntxWeightOnlyConfig(
        weight_dtype={"4w": torch.int4, "8w": torch.int8}[bits],
        granularity=PerAxis(0) if gran == "pc" else PerGroup(int(gran[1:])),
        mapping_type=MappingType.SYMMETRIC,
    )


def _load_wav(path: str) -> torch.Tensor:
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1, path
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0)


def main() -> None:
    from transformers import AutoTokenizer
    from torchao.quantization import quantize_

    clips, recipes = sys.argv[1].split(","), sys.argv[2:]
    dtype = {"bf16": torch.bfloat16}.get(os.environ.get("DTYPE"), torch.float32)
    tok = AutoTokenizer.from_pretrained(q.model.SOURCE_REPO)
    prefix, suffix = tok(q.exportable.PROMPT_PREFIX).input_ids, tok(q.exportable.PROMPT_SUFFIX).input_ids
    language = tok("language").input_ids[0]

    m = q.Qwen3Asr()
    pre, enc, emb = m.preprocessor().eval(), m.audio_encoder().eval(), m.token_embedding()
    prompts = []
    with torch.no_grad():
        for clip in clips:
            mel = pre(_load_wav(clip))
            pieces = []
            for start in range(0, mel.shape[2], q.WINDOW_FRAMES):
                window = mel[:, :, start : start + q.WINDOW_FRAMES]
                frames = window.shape[2]
                pad = math.ceil(frames / q.CHUNK_FRAMES) * q.CHUNK_FRAMES - frames
                n = q.audio_token_count(frames)
                pieces.append(enc(torch.nn.functional.pad(window, (0, pad)), torch.tensor([n]))[0, :n])
            prompts.append(
                torch.cat([emb(torch.tensor([prefix])), torch.cat(pieces)[None], emb(torch.tensor([suffix]))], 1)
            )

    for recipe in recipes:
        rules = [tuple(rule.split("=")) for rule in recipe.split(";")]

        def scheme(fqn: str) -> str:
            for pattern, s in rules:
                if pattern == "default" or re.search(pattern, fqn):
                    return s
            return "none"

        text = m.text_model(use_kv_cache=False, max_seq_len=q.MAX_CONTEXT_LEN, max_context_len=q.MAX_CONTEXT_LEN)
        text = text.eval().to(dtype)
        for s in sorted({s for _, s in rules} - {"none"}):
            quantize_(
                text,
                _config(s),
                filter_fn=lambda mod, fqn, s=s: isinstance(mod, torch.nn.Linear) and scheme(fqn) == s,
            )
        with torch.no_grad():
            logp = [
                float(text(None, None, x.to(dtype)).float().reshape(-1)[-text.vocab_size :].log_softmax(-1)[language])
                for x in prompts
            ]
        print(f"{recipe:60s} min {min(logp):6.2f} mean {sum(logp) / len(logp):6.2f}  " + " ".join(f"{v:.2f}" for v in logp))


if __name__ == "__main__":
    main()
