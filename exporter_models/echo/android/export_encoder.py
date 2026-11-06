"""The demo app's reference encoder: Echo's codec encoder on XNNPACK with one method per whole
second, encode_3s ... encode_10s (wav f32[s * 24000] -> codes i64[8, s * 25]). The app rounds a
recording up to the next second, so a short reference is not padded to 10 s of silence. Methods
share one weight set in the program (~650 MB).

usage: python export_encoder.py <out.pte>   (the uv project of exporter_models/echo)
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _echo import hub  # noqa: E402
from _echo.codec import Encoder, load_codec  # noqa: E402

SECONDS = range(3, 11)


def main():
    out = sys.argv[1]
    torch.set_grad_enabled(False)
    codec = load_codec(hub.codec())
    enc = Encoder(codec).eval()
    hop, rate = codec.config.hop_length, codec.config.sample_rate
    eps = {}
    for s in SECONDS:
        n = s * rate // hop * hop
        wav = torch.randn(n, generator=torch.Generator().manual_seed(s)) * 0.05
        assert torch.equal(enc(wav), codec.encode(wav.view(1, 1, -1)).audio_codes[0]), s
        eps[f"encode_{s}s"] = torch.export.export(enc, (wav,), strict=False)

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    t0 = time.time()
    edge = to_edge_transform_and_lower(eps, partitioner=[XnnpackPartitioner()],
                                       compile_config=EdgeCompileConfig(_check_ir_validity=False))
    buf = edge.to_executorch().buffer
    Path(out).write_bytes(buf)
    print(f"wrote {out} ({len(buf) / 2**20:.0f} MiB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
