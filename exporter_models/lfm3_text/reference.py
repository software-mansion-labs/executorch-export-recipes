"""Save Hugging Face reference logits for LFM3 parity checks.

Runs the upstream remote-code model (bf16 or fp32, eager attention) on a chat prompt
long enough to cross the 256-token sliding window, then greedy-decodes a few
tokens. Run it in its own process: it and the exported model do not fit in
RAM together on a 16-18 GB Mac.

    python exporter_models/lfm3_text/reference.py --source <LFM3-1.7B snapshot> --output ref.pt
"""

import argparse

import torch

PROMPT = (
    "Summarise the following notes in three bullet points, then answer the question at the end.\n\n"
    + " ".join(
        f"Note {i}: the {w} sensor on rig {i % 7} reported {17 * i % 101} units at hour {i % 24}."
        for i, w in enumerate(["thermal", "pressure", "acoustic", "optical", "strain"] * 6)
    )
    + "\n\nQuestion: which rig appears most often, and what is the capital of France?"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--new-tokens", type=int, default=24)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.source, trust_remote_code=True, dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        attn_implementation="eager"
    ).eval()
    ids = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    )["input_ids"]
    with torch.no_grad():
        prefill = model(ids).logits[0].float()
        # Plain greedy: the checkpoint's generation_config sets repetition_penalty
        # 1.1, which generate() would otherwise apply on top of do_sample=False.
        out = model.generate(
            ids, max_new_tokens=args.new_tokens, do_sample=False, repetition_penalty=1.0
        )
    gen = out[0, ids.shape[1]:]
    print(f"prompt tokens: {ids.shape[1]}")
    print("greedy:", repr(tok.decode(gen)))
    torch.save({"input_ids": ids[0], "prefill_logits": prefill, "generated": gen}, args.output)


if __name__ == "__main__":
    main()
