from executorch.runtime import Runtime
from tokenizers import Tokenizer
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import requests
import io
import random
import ctypes
import os

def main():
    parser = argparse.ArgumentParser(description="Whisper Audio Transcription Tool (Bundled Model)")
    parser.add_argument("--model", type=str, required=True, help="Path to bundled .pte model")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json")
    parser.add_argument("--input", type=str, required=True, help="URL or local path to audio file")
    args = parser.parse_args()

    # Load quantized/custom operators via dynamic linking to resolve missing kernels
    libs_to_load = [
        "/Users/igorswat/Documents/Projekty/executorch/.venv/lib/python3.12/site-packages/executorch/kernels/quantized/libquantized_ops_aot_lib.dylib"
    ]
    for lib_path in libs_to_load:
        if os.path.exists(lib_path):
            print(f"Loading custom operators from {lib_path}...")
            # Use RTLD_GLOBAL to ensure symbols are available to the ExecuTorch runtime
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        else:
            print(f"Warning: Operator library not found at: {lib_path}")

    print(f"\n[1/4] Processing audio from: {args.input}...")
    
    if args.input.startswith(("http://", "https://")):
        response = requests.get(args.input, timeout=10)
        waveform, sample_rate = torchaudio.load(io.BytesIO(response.content))
    else:
        waveform, sample_rate = torchaudio.load(args.input)

    # Resample to 16kHz
    if sample_rate != 16000:
        print("Resampling...")
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # The bundled model preprocessor handles padding/trimming up to 30s.
    duration_s = 30
    duration = int(duration_s * 16000)
    
    audio_tensor = waveform.squeeze()[:duration].float()

    runtime = Runtime.get()
    print(f"[2/4] Loading ExecuTorch program '{args.model}'...")
    program = runtime.load_program(args.model)
    
    # Load separate methods from the bundled program
    encode_method = program.load_method("encode")
    decode_method = program.load_method("decode")

    import time
    from math import ceil
    # ANSI color codes
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    print(f"{CYAN}[3/4] Running Encoder...{RESET}")
    start_encode = time.perf_counter()
    encodings = encode_method.execute([audio_tensor])[0]
    encode_time = (time.perf_counter() - start_encode) * 1000

    print(f"{CYAN}[4/4] Transcribing...{RESET}")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    tokens = [50257, 50363] # for .en models
    current_tokens = torch.tensor([tokens], dtype=torch.long)
    current_positions = torch.tensor([0, 1], dtype=torch.long)

    print("\n" + BOLD + "="*50 + RESET)
    print(BOLD + "TRANSCRIPTION RESULTS" + RESET)
    print(BOLD + "="*50 + RESET)
    print(BOLD + "Text Output: " + RESET, end="", flush=True)

    total_decode_time = 0.0
    steps = 0
    for i in range(126): # Safety limit for generation
        step_start = time.perf_counter()
        outputs = decode_method.execute([current_tokens, current_positions, encodings])
        step_time = (time.perf_counter() - step_start) * 1000
        total_decode_time += step_time
        steps += 1
        # Get logits (assuming shape [1, seq_len, vocab])
        logits = outputs[0]
        next_token_logits = logits[0, -1, :]
        # Softmax and greedy decode
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(probs).item()
        # Print decoded token
        decoded_text = tokenizer.decode([next_token], skip_special_tokens=True)
        print(decoded_text, end="", flush=True)
        if next_token == 50256: # End of text token
            break
        # Update state for next step: pass only the new token
        tokens.append(next_token)
        current_tokens = torch.tensor([[next_token]], dtype=torch.long)
        current_positions = torch.tensor([len(tokens) - 1], dtype=torch.long)

    print("\n\n" + BOLD + "="*50 + RESET)
    print(f"{GREEN}{BOLD}TIMING SUMMARY{RESET}")
    print(f"{YELLOW}Encoder time:   {encode_time:.2f} ms ({encode_time/1000:.2f} s){RESET}")
    print(f"{YELLOW}Decoder time:   {total_decode_time:.2f} ms for {steps} steps (avg: {total_decode_time/steps:.2f} ms/step){RESET}")
    print(f"{YELLOW}Total time:     {encode_time + total_decode_time:.2f} ms ({(encode_time + total_decode_time)/1000:.2f} s){RESET}")
    print(BOLD + "="*50 + RESET + "\n")

if __name__ == "__main__":
    main()