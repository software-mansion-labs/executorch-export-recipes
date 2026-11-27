"""Write the fixed seed-42 inputs consumed by native_probe.cpp and validate.py."""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    args=parser.parse_args()
    text=args.output/'text';text.mkdir(parents=True,exist_ok=True)
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    prompt='<|startoftext|><|im_start|>system\nRespond in text.<|im_end|>\n<|im_start|>user\nWhat is 2 + 2? Answer briefly.<|im_end|>\n<|im_start|>assistant\n<think></think>'
    (text/'prompt.txt').write_text(prompt)
    (text/'prompt.bin').write_bytes(torch.tensor(tokenizer.encode(prompt).ids,dtype=torch.int64).numpy().tobytes())
    out=args.output/'components';out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    values={'depth-hidden':torch.randn(1,2048),'depth-codes':torch.randint(0,2016,(1,12)),'depth-mask':(torch.arange(12)[None]%2==0).byte()}
    for name,value in values.items():(out/(name+'.bin')).write_bytes(value.contiguous().numpy().tobytes())
    for count in [1,4,32]:(out/f'codec-{count}.bin').write_bytes(torch.randint(0,2016,(1,12,count)).numpy().tobytes())
    for count in [400,16400,80400]:(out/f'wave-{count}.bin').write_bytes((torch.randn(1,count)*.1).numpy().tobytes())


if __name__=='__main__':
    main()
