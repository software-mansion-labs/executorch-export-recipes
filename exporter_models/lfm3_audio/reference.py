"""Save Hugging Face reference outputs for LFM3-Audio (audio in, text out).

ASR on one 16 kHz mono clip with the model card's prompt, greedy without the
checkpoint's repetition penalty, speech output suppressed. Also saves the
audio features (`get_audio_features`) and the padded waveform the exported
`audio_encoder` takes. Run in its own process (RAM).

    python exporter_models/lfm3_audio/reference.py --source <LFM3-Audio-2B snapshot> --audio clip_16k.wav --output ref.pt
"""

import argparse
import wave

import numpy as np
import torch


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    audio = read_wav(args.audio)
    proc = AutoProcessor.from_pretrained(args.source, trust_remote_code=True)
    conversation = [
        {"role": "system", "content": "Perform ASR in English."},
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]
    inputs = proc.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        sampling_rate=16000,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.source, trust_remote_code=True,
        dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16,
    ).eval()
    with torch.no_grad():
        feats = model.get_audio_features(inputs["input_features"], inputs["feature_attention_mask"])
        prefill = model(**inputs).logits[0].float()
        out = model.generate(
            **inputs, max_new_tokens=args.new_tokens, do_sample=False, repetition_penalty=1.0,
            suppress_mixed=True, suppress_audio_out=True, return_dict_in_generate=True,
        )
    ids = inputs["input_ids"][0]
    gen = out.sequences[0, ids.shape[0]:]
    fe = proc.feature_extractor
    padded = torch.nn.functional.pad(torch.from_numpy(audio), (0, fe.pad_length(len(audio)) - len(audio)))
    audio_id = proc.audio_token_id
    print(f"prompt tokens: {ids.shape[0]} (audio tokens: {(ids == audio_id).sum().item()})")
    print("greedy:", repr(proc.decode(gen, skip_special_tokens=True)))
    torch.save(
        {
            "input_ids": ids,
            "audio_token_id": audio_id,
            "waveform": padded[None].contiguous(),
            "audio_features": feats.float(),
            "prefill_logits": prefill,
            "generated": gen,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
