"""Check the exportable decoder (eager, before lowering) against `reference.py`.

Prefills the reference prompt in chunks (so the KV caches, conv states and the
sliding-window mask are all exercised across calls), compares the logits at
every chunk boundary with the HF prefill logits, then greedy-decodes and
compares the tokens.

    python exporter_models/lfm3_text/parity.py --source <snapshot> --reference ref.pt [--attn-impl eager|llama]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lfm3.load import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--attn-impl", default="eager")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--window-slice", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()

    if args.attn_impl == "llama":
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    ref = torch.load(args.reference)
    model = build(Path(args.source), dtype=dtype, max_context_len=1024, attn_impl=args.attn_impl,
        window_slice=args.window_slice,
    )
    ids = ref["input_ids"][None]
    n = ids.shape[1]

    worst = 1.0
    with torch.no_grad():
        pos = 0
        while pos < n:
            t = ids[:, pos : pos + args.chunk]
            logits = model(t, torch.tensor([pos])).float()[0]
            pos += t.shape[1]
            want = ref["prefill_logits"][pos - 1]
            cos = torch.nn.functional.cosine_similarity(logits, want, dim=0).item()
            worst = min(worst, cos)
            print(f"pos {pos:4d}: cos {cos:.5f} argmax {logits.argmax().item()} vs {want.argmax().item()}")
        # Teacher-forced decode: feed HF's greedy tokens one at a time, so a
        # single near-tie flip does not derail the rest of the comparison.
        want = ref["generated"].tolist()
        gen = [logits.argmax().item()]
        top2 = []
        for tok in want[:-1]:
            logits = model(torch.tensor([[tok]]), torch.tensor([pos]))[0].float()
            pos += 1
            gen.append(logits.argmax().item())
            v = logits.topk(2).values
            top2.append(round((v[0] - v[1]).item(), 3))
    match = sum(a == b for a, b in zip(gen, want))
    print(f"worst prefill cos {worst:.5f}; teacher-forced argmax matching HF greedy: {match}/{len(want)}")
    for i, (a, b) in enumerate(zip(gen, want)):
        if a != b:
            print(f"  step {i}: ours {a} hf {b} (our top-2 logit margin {top2[i - 1] if i else 'n/a'})")


if __name__ == "__main__":
    main()
