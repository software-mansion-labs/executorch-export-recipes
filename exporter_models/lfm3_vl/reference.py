"""Save Hugging Face reference outputs for LFM3-VL parity checks.

The image is resized to 512x512 first and the processor is told not to resize,
so the reference sees exactly what RNE's runner feeds `vision_encoder`
(RNE letterboxes to 512x512 on device). Greedy decode without the checkpoint's
repetition penalty. Run in its own process (RAM).

    python exporter_models/lfm3_vl/reference.py --source <LFM3-VL-2B snapshot> --image img.jpg --output ref.pt
"""

import argparse

import numpy as np
import torch

PROMPT = "Describe this image in one sentence."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--new-tokens", type=int, default=24)
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"])
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    img = Image.open(args.image).convert("RGB").resize((512, 512), Image.BILINEAR)
    proc = AutoProcessor.from_pretrained(args.source, trust_remote_code=True)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    inputs = proc(text=text, images=[img], images_kwargs={"do_resize": False}, return_tensors="pt")
    model = AutoModelForImageTextToText.from_pretrained(
        args.source, trust_remote_code=True,
        dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    with torch.no_grad():
        prefill = model(**inputs).logits[0].float()
        out = model.generate(
            **inputs, max_new_tokens=args.new_tokens, do_sample=False, repetition_penalty=1.0
        )
    ids = inputs["input_ids"][0]
    gen = out[0, ids.shape[0]:]
    print(f"prompt tokens: {ids.shape[0]} (image tokens: {(ids == model.config.image_token_id).sum().item()})")
    print("greedy:", repr(proc.tokenizer.decode(gen)))
    torch.save(
        {
            "input_ids": ids,
            "image_token_id": model.config.image_token_id,
            "image": torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().contiguous()[None],
            "prefill_logits": prefill,
            "generated": gen,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
