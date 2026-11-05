"""Run an exported LFM3 text .pte against `reference.py` output.

Checks the baked contract (constant methods, get_model_schema), prefills the
reference prompt in chunks of the schema's seqLen max, compares logits at each
chunk boundary, then teacher-forces the reference's greedy tokens and counts
argmax agreement. Also times prefill and decode (a smoke test, not a
benchmark: host CPU, one run).

    python exporter_models/lfm3_text/validate.py --model lfm_3_1_7b_xnnpack_8da4w.pte --reference ref.pt
"""

import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()

    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    from executorch.kernels import quantized  # noqa: F401
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(args.model)
    methods = sorted(program.method_names)
    consts = {}
    for name in methods:
        if name == "forward":
            continue
        out = program.load_method(name).execute([])
        consts[name] = out[0] if len(out) == 1 else list(out)
    schema = json.loads(consts.pop("get_model_schema"))
    seq_max = schema["forward"]["inputs"][0]["shape"][1]["range"]["max"]
    print("constants:", consts)
    print("forward seqLen range max:", seq_max)

    fwd = program.load_method("forward")
    ref = torch.load(args.reference)
    ids = ref["input_ids"][None]
    n = ids.shape[1]
    worst, pos, t0 = 1.0, 0, time.perf_counter()
    while pos < n:
        t = ids[:, pos : pos + seq_max].contiguous()
        logits = fwd.execute([t, torch.tensor([pos])])[0][0].float()
        pos += t.shape[1]
        want = ref["prefill_logits"][pos - 1]
        cos = torch.nn.functional.cosine_similarity(logits, want, dim=0).item()
        worst = min(worst, cos)
        print(f"pos {pos:4d}: cos {cos:.5f} argmax {logits.argmax().item()} vs {want.argmax().item()}")
    prefill_s = time.perf_counter() - t0

    want = ref["generated"].tolist()
    gen = [logits.argmax().item()]
    t0 = time.perf_counter()
    for tok in want[:-1]:
        logits = fwd.execute([torch.tensor([[tok]]), torch.tensor([pos])])[0][0]
        pos += 1
        gen.append(logits.argmax().item())
    decode_s = time.perf_counter() - t0
    match = sum(a == b for a, b in zip(gen, want))
    print(f"worst prefill cos {worst:.5f}; teacher-forced argmax = HF greedy: {match}/{len(want)}")
    print(f"host timing: prefill {n / prefill_s:.1f} tok/s, decode {(len(want) - 1) / decode_s:.1f} tok/s")


if __name__ == "__main__":
    main()
