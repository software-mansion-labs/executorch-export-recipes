"""Compare the exportable `AudioEncoder` (waveform in) with the upstream
`get_audio_features` saved by `reference.py` (log-mel features in).

    python exporter_models/lfm3_audio/audio_parity.py --source <LFM3-Audio-2B snapshot> --reference ref.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lfm3.audio import AudioEncoder, EncoderConfig, convert, mel_filters  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()

    from safetensors.torch import load_file

    hf = json.loads((Path(args.source) / "config.json").read_text())
    enc = AudioEncoder(hf, mel_filters(), max_tokens=1024)
    state = convert(load_file(str(Path(args.source) / "model.safetensors")), EncoderConfig.from_hf(hf))
    missing, unexpected = enc.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    ref = torch.load(args.reference)
    with torch.no_grad():
        got = enc.eval()(ref["waveform"])[0]
    want = ref["audio_features"]
    cos = torch.nn.functional.cosine_similarity(got, want, dim=-1)
    print(f"tokens {tuple(got.shape)} vs {tuple(want.shape)}")
    print(f"per-token cos min {cos.min().item():.6f} mean {cos.mean().item():.6f}; "
          f"max abs err {(got - want).abs().max().item():.4g} (|ref| max {want.abs().max().item():.4g})")


if __name__ == "__main__":
    main()
