"""Run LFM3-VL (an exported .pte, or the eager vendored modules) against
`reference.py` output, the way RNE's multimodal runner drives it.

The image-token run of the reference prompt is replaced by `vision_encoder`
output; everything else goes through `token_embedding`. The embeddings are
prefilled through `text_decoder` in chunks of the schema's seqLen max, logits
are compared at each chunk boundary, then the reference's greedy tokens are
teacher-forced.

    python exporter_models/lfm3_vl/validate.py --model lfm_3_vl_2b_xnnpack_8da4w.pte --reference ref.pt
    python exporter_models/lfm3_vl/validate.py --eager --source <snapshot> --reference ref.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


class PteRunner:
    def __init__(self, path):
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
        from executorch.kernels import quantized  # noqa: F401
        from executorch.runtime import Runtime

        self.program = Runtime.get().load_program(path)
        schema = json.loads(self.program.load_method("get_model_schema").execute([])[0])
        self.seq_max = schema["text_decoder"]["inputs"][0]["shape"][1]["range"]["max"]
        self.m = {n: self.program.load_method(n) for n in ("vision_encoder", "token_embedding", "text_decoder")}

    def vision(self, img):
        return self.m["vision_encoder"].execute([img])[0]

    def embed(self, ids):
        return self.m["token_embedding"].execute([ids.contiguous()])[0]

    def decode(self, emb, pos):
        return self.m["text_decoder"].execute([emb.contiguous(), pos])[0][0].float()


class EagerRunner:
    def __init__(self, source):
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

        from _lfm3.export_lib import TextDecoder, TokenEmbedding, build_all

        self.vis, lm = build_all(Path(source), "xnnpack", 1024)
        self.emb_m, self.dec_m = TokenEmbedding(lm.embed_tokens), TextDecoder(lm)
        self.seq_max = 256

    @torch.no_grad()
    def vision(self, img):
        return self.vis(img)

    @torch.no_grad()
    def embed(self, ids):
        return self.emb_m(ids)

    @torch.no_grad()
    def decode(self, emb, pos):
        return self.dec_m(emb, pos)[0].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--source")
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()

    r = EagerRunner(args.source) if args.eager else PteRunner(args.model)
    ref = torch.load(args.reference)
    ids = ref["input_ids"]
    is_img = ids == ref["image_token_id"]
    first = int(is_img.nonzero()[0])
    n_img = int(is_img.sum())

    t0 = time.perf_counter()
    img_emb = r.vision(ref["image"])
    vision_s = time.perf_counter() - t0
    assert img_emb.shape[1] == n_img, (img_emb.shape, n_img)
    segments = [
        r.embed(ids[None, :first]),
        img_emb,
        r.embed(ids[None, first + n_img :]),
    ]
    embeds = torch.cat(segments, dim=1)

    worst, pos, n = 1.0, 0, embeds.shape[1]
    t0 = time.perf_counter()
    while pos < n:
        chunk = embeds[:, pos : pos + r.seq_max]
        T = chunk.shape[1]
        logits = r.decode(chunk, torch.arange(pos, pos + T))
        pos += T
        want = ref["prefill_logits"][pos - 1]
        cos = torch.nn.functional.cosine_similarity(logits, want, dim=0).item()
        worst = min(worst, cos)
        print(f"pos {pos:4d}: cos {cos:.5f} argmax {logits.argmax().item()} vs {want.argmax().item()}")
    prefill_s = time.perf_counter() - t0

    want = ref["generated"].tolist()
    gen = [logits.argmax().item()]
    t0 = time.perf_counter()
    for tok in want[:-1]:
        logits = r.decode(r.embed(torch.tensor([[tok]])), torch.tensor([pos]))
        pos += 1
        gen.append(logits.argmax().item())
    decode_s = time.perf_counter() - t0
    match = sum(a == b for a, b in zip(gen, want))
    print(f"worst prefill cos {worst:.5f}; teacher-forced argmax = HF greedy: {match}/{len(want)}")
    print(f"host timing: vision {vision_s * 1000:.0f} ms, prefill {n / prefill_s:.1f} tok/s, "
          f"decode {(len(want) - 1) / decode_s:.1f} tok/s")


if __name__ == "__main__":
    main()
