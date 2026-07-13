#!/usr/bin/env python3
"""Benchmark the exported ExecuTorch XNNPACK Supertonic 3 model.

Loads the four ``.pte`` files produced by ``scripts/export/export_xnnpack.py``
through the ExecuTorch runtime and drives the full TTS pipeline (text
preprocessing, the flow-matching diffusion loop and vocoding on the host).
Reports per-stage and end-to-end timing / real-time factor, and optionally
verifies the audio against the eager PyTorch pipeline for the same noise.

Usage:
    python scripts/benchmark/benchmark_xnnpack.py --model-pte-dir checkpoints/xnnpack
    python scripts/benchmark/benchmark_xnnpack.py --model-pte-dir checkpoints/xnnpack --verify
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from supertonic_torch.et_pipeline import ExecuTorchTTS
from supertonic_torch.pipeline import DEFAULT_SPEED

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world! This is a text-to-speech synthesis test.",
    "Artificial intelligence is transforming how we interact with technology every day.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-pte-dir", default=str(_ROOT / "checkpoints" / "xnnpack"),
                        help="Directory with the four exported .pte files")
    parser.add_argument("--model-dir", default=str(_ROOT / "onnx"),
                        help="Directory with tts.json + unicode_indexer.json")
    parser.add_argument("--checkpoint", default=str(_ROOT / "checkpoints" / "supertonic3.pt"),
                        help="Eager PyTorch checkpoint (used only with --verify)")
    parser.add_argument("--voice-style-dir", default=str(_ROOT / "voice_styles"))
    parser.add_argument("--voice", default="M1")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="benchmark_output")
    parser.add_argument("--verify", action="store_true",
                        help="Compare audio against eager PyTorch (same noise)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ExecuTorch programs from: {args.model_pte_dir}")
    t0 = time.perf_counter()
    pipe = ExecuTorchTTS(args.model_dir, args.model_pte_dir, args.voice_style_dir)
    print(f"Loaded in {time.perf_counter() - t0:.2f}s")
    style = pipe.get_voice_style(args.voice)

    print("Warmup...")
    pipe.synthesize("Warmup.", style, total_steps=args.steps, speed=args.speed,
                    lang=args.lang, generator=torch.Generator().manual_seed(args.seed))

    print(f"\n{'='*72}")
    print(f"ExecuTorch XNNPACK benchmark  (voice={args.voice}, lang={args.lang}, "
          f"steps={args.steps}, repeat={args.repeat})")
    print(f"{'='*72}")
    print(f"{'text':<52} | {'time':>7} | {'audio':>6} | {'RTF':>6}")
    print("-" * 72)

    all_rtf, all_t = [], []
    for text in SAMPLE_TEXTS:
        times = []
        for _ in range(args.repeat):
            gen = torch.Generator().manual_seed(args.seed)
            t = time.perf_counter()
            wav, dur = pipe.synthesize(text, style, total_steps=args.steps,
                                       speed=args.speed, lang=args.lang, generator=gen)
            times.append(time.perf_counter() - t)
        avg = float(np.mean(times))
        audio_s = wav.shape[1] / pipe.sample_rate
        rtf = avg / audio_s
        all_rtf.append(rtf)
        all_t.append(avg)
        print(f"{text[:50]:<52} | {avg:6.3f}s | {audio_s:5.2f}s | {rtf:6.3f}")
        stages = " ".join(f"{k}={v*1000:.0f}ms" for k, v in pipe.timings.items())
        print(f"    stages(last run): {stages}")

        import soundfile as sf
        sf.write(str(out_dir / f"et_{args.voice}_{SAMPLE_TEXTS.index(text):02d}.wav"),
                 wav.squeeze(), pipe.sample_rate)

    print("-" * 72)
    print(f"Mean time: {np.mean(all_t):.3f}s | Mean RTF: {np.mean(all_rtf):.3f}")

    if args.verify:
        print(f"\n{'='*72}\nVerifying against eager PyTorch (same noise)\n{'='*72}")
        from supertonic_torch import TTS
        tts = TTS(model_dir=args.model_dir, checkpoint=args.checkpoint,
                  voice_style_dir=args.voice_style_dir)
        estyle = tts.get_voice_style(args.voice)
        for text in SAMPLE_TEXTS:
            g1 = torch.Generator().manual_seed(args.seed)
            g2 = torch.Generator().manual_seed(args.seed)
            t = time.perf_counter()
            wav_e, _ = tts.synthesize(text, estyle, total_steps=args.steps, speed=args.speed,
                                      lang=args.lang, generator=g2)
            t_eager = time.perf_counter() - t
            wav_x, _ = pipe.synthesize(text, style, total_steps=args.steps, speed=args.speed,
                                       lang=args.lang, generator=g1)
            n = min(wav_e.shape[1], wav_x.shape[1])
            diff = np.abs(wav_e[:, :n] - wav_x[:, :n]).max()
            peak = np.abs(wav_e).max()
            print(f"  '{text[:40]}...' eager={t_eager:.3f}s  max_abs_diff={diff:.4e} "
                  f"(peak {peak:.3f}, rel {diff/peak:.2%})")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
