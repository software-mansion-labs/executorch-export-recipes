"""Save Hugging Face logits separately from the memory-intensive export/run."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1)
args = parser.parse_args()
torch.set_num_threads(4)
source = args.source
args.output.parent.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(source)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "What is the capital of France? Reply with just the city.",
    },
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
ids = tokenizer.encode(prompt, add_special_tokens=False)
print("PROMPT", repr(prompt), "ids", ids, flush=True)
model = AutoModelForCausalLM.from_pretrained(
    source, dtype=torch.bfloat16, attn_implementation="eager"
)
model.eval()
steps = []
logits = []
past = None
tokens = torch.tensor([ids])
with torch.inference_mode():
    for step in range(args.steps):
        result = model(tokens, past_key_values=past, use_cache=True)
        last = result.logits[0, -1].float().cpu()
        next_id = int(last.argmax())
        logits.append(last.clone())
        steps.append(next_id)
        print("STEP", step, next_id, repr(tokenizer.decode([next_id])), flush=True)
        past = result.past_key_values
        tokens = torch.tensor([[next_id]])
        if next_id == 124900:
            break
torch.save(
    {
        "prompt": prompt,
        "input_ids": ids,
        "tokens": steps,
        "logits": torch.stack(logits),
    },
    args.output,
)
print("REFERENCE", str(source), repr(tokenizer.decode(steps)), flush=True)
