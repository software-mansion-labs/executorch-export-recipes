"""
Test script for CoreML-exported Whisper .pte models.

The decoder was exported with a STATIC shape [1, 128] for input_ids
(CoreML doesn't support dynamic shapes). Decoding must therefore:
  - pre-allocate a [1, 128] tensor padded with zeros
  - place new tokens in-place at step N
  - read logits from position N (not the last position)

Usage:
    python3 test_coreml_whisper.py                    # synthetic audio (sanity check)
    python3 test_coreml_whisper.py path/to/audio.wav  # real audio file
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoProcessor
from executorch.runtime import Runtime

ENCODER_PATH = "whisper_encoder_coreml.pte"
DECODER_PATH = "whisper_decoder_coreml.pte"
MODEL_NAME = "openai/whisper-tiny.en"
SR = 16_000
MAX_SEQ_LEN = 128
BOS_TOKEN = 50257  # <|startoftranscript|>
EOS_TOKEN = 50256  # <|endoftext|>


class ETModule:
    def __init__(self, path: str):
        rt = Runtime.get()
        program = rt.load_program(Path(path))
        self._method = program.load_method("forward")

    def __call__(self, *args: torch.Tensor) -> list[torch.Tensor]:
        return self._method.execute(args)


def load_audio(path: str) -> torch.Tensor:
    """Load audio, trying multiple backends."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path, backend="soundfile")
        if sr != SR:
            waveform = torchaudio.functional.resample(waveform, sr, SR)
        return waveform[0]
    except Exception:
        pass
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        audio = torch.from_numpy(data if data.ndim == 1 else data[:, 0])
        if sr != SR:
            import torchaudio
            audio = torchaudio.functional.resample(audio.unsqueeze(0), sr, SR).squeeze(0)
        return audio
    except Exception as e:
        raise RuntimeError(f"Could not load audio from {path}: {e}")


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Loading tokenizer from {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer

    print(f"Loading encoder from {ENCODER_PATH}...")
    t0 = time.perf_counter()
    encoder = ETModule(ENCODER_PATH)
    print(f"  Encoder loaded in {(time.perf_counter() - t0)*1000:.0f} ms")

    print(f"Loading decoder from {DECODER_PATH}...")
    t0 = time.perf_counter()
    decoder = ETModule(DECODER_PATH)
    print(f"  Decoder loaded in {(time.perf_counter() - t0)*1000:.0f} ms")

    CHUNK_SAMPLES = 29 * SR   # 464 000 — matches C++ kChunkSize
    MIN_CHUNK_SAMPLES = SR    # 1 s minimum, matches C++ kMinChunkSamples

    if audio_path:
        print(f"\nLoading audio from {audio_path}...")
        audio = load_audio(audio_path)
        print(f"  {len(audio) / SR:.2f}s of audio at {SR} Hz")
    else:
        print("\nNo audio file provided — using 3s of synthetic noise")
        audio = torch.randn(SR * 3) * 0.01

    all_text_parts = []
    chunk_starts = range(0, len(audio), CHUNK_SAMPLES)
    num_chunks = len(chunk_starts)

    for chunk_idx, start in enumerate(chunk_starts):
        chunk = audio[start:start + CHUNK_SAMPLES]
        if len(chunk) < MIN_CHUNK_SAMPLES:
            break

        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} "
              f"({start / SR:.1f}s – {(start + len(chunk)) / SR:.1f}s) ---")

        # --- Encode ---
        print("[Encoder] Running forward...")
        t0 = time.perf_counter()
        encoder_output = encoder(chunk)[0]  # [1, 1500, 384]
        encode_ms = (time.perf_counter() - t0) * 1000
        print(f"  Output shape: {encoder_output.shape}  ({encode_ms:.0f} ms)")

        # --- Decode (greedy, static-shape) ---
        # The decoder was exported with static input shape [1, MAX_SEQ_LEN].
        # We pre-fill with zeros and place tokens in-place.
        # At step N: input_ids has real tokens at [0..N], zeros at [N+1..127].
        # Causal attention ensures logits[0, N, :] is unaffected by future zeros.
        print("[Decoder] Running greedy decode (static shape)...")
        input_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64)
        input_ids[0, 0] = BOS_TOKEN

        total_decode_ms = 0.0
        final_step = 0

        for step in range(MAX_SEQ_LEN - 1):
            t0 = time.perf_counter()
            logits = decoder(input_ids, encoder_output)[0]  # [1, 128, vocab_size]
            total_decode_ms += (time.perf_counter() - t0) * 1000

            # Take the prediction at the current position
            next_token = int(torch.argmax(logits[0, step, :]).item())
            final_step = step + 1

            if next_token == EOS_TOKEN:
                print(f"  EOS reached at step {step + 1}")
                break

            input_ids[0, step + 1] = next_token

            if (step + 1) % 10 == 0:
                partial = tokenizer.decode(
                    input_ids[0, :step + 2].tolist(), skip_special_tokens=True
                )
                print(f"  Step {step + 1:3d}: {partial!r}")
        else:
            print(f"  Reached MAX_SEQ_LEN ({MAX_SEQ_LEN})")

        avg_decode_ms = total_decode_ms / max(final_step, 1)
        print(f"  Avg per-step: {avg_decode_ms:.1f} ms  "
              f"(total {total_decode_ms:.0f} ms, {final_step} steps)")

        generated_ids = input_ids[0, :final_step].tolist()
        chunk_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"  Chunk transcription: {chunk_text!r}")
        if chunk_text.strip():
            all_text_parts.append(chunk_text.strip())

    text = " ".join(all_text_parts)
    print(f"\nFull transcription: {text!r}")


if __name__ == "__main__":
    main()
