import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import json
import time
import ctypes
import re
from executorch.runtime import Runtime
from tokenizers import Tokenizer

def normalize_text(text):
    """Normalize text for comparison: lowercase and remove punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def calculate_wer(reference, hypothesis):
    """
    Calculation of Word Error Rate (WER) using Levenshtein distance.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    
    # Initialize matrix
    d = [[0 for _ in range(len(hyp_words) + 1)] for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1): d[i][0] = i
    for j in range(len(hyp_words) + 1): d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
                
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

def main():
    parser = argparse.ArgumentParser(description="Multi-file Whisper Evaluation Tool")
    parser.add_argument("--model", type=str, required=True, help="Path to bundled .pte model")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing sample_i.json and sample_i.wav files")
    args = parser.parse_args()

    # Load quantized/custom operators via dynamic linking to resolve missing kernels
    libs_to_load = [
        "/Users/igorswat/Documents/Projekty/executorch/.venv/lib/python3.12/site-packages/executorch/kernels/quantized/libquantized_ops_aot_lib.dylib"
    ]
    for lib_path in libs_to_load:
        if os.path.exists(lib_path):
            print(f"Loading custom operators from {lib_path}...")
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        else:
            print(f"Warning: Operator library not found at: {lib_path}")

    # ANSI color codes
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    print(f"\n{BOLD}Loading Runtime and Tokenizer...{RESET}")
    runtime = Runtime.get()
    program = runtime.load_program(args.model)
    encode_method = program.load_method("encode")
    decode_method = program.load_method("decode")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Find all sample files
    base_names = []
    for file in os.listdir(args.data_dir):
        if file.endswith(".json") and file.startswith("sample_"):
            base_names.append(file.replace(".json", ""))
    
    base_names = sorted(base_names, key=lambda x: int(x.split('_')[1]))
    
    results = []
    
    print(f"\n{BOLD}Starting Evaluation on {len(base_names)} samples from: {args.data_dir}{RESET}\n")
    print(f"{'Sample':<12} | {'WER':<8} | {'Time (s)':<10} | {'Transcription'}")
    print("-" * 80)

    for base in base_names:
        json_path = os.path.join(args.data_dir, f"{base}.json")
        wav_path = os.path.join(args.data_dir, f"{base}.wav")
        
        if not os.path.exists(wav_path):
            continue
            
        with open(json_path, 'r') as f:
            meta = json.load(f)
            reference_text = meta['transcription']

        # [1] Process Audio
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        duration = int(30 * 16000)
        audio_tensor = waveform.squeeze()[:duration].float()

        # [2] Run Inference
        start_time = time.perf_counter()
        
        # Encoder
        encodings = encode_method.execute([audio_tensor])[0]
        
        # Decoder (Greedy)
        tokens = [50257, 50363]
        current_tokens = torch.tensor([tokens], dtype=torch.long)
        current_positions = torch.tensor([0, 1], dtype=torch.long)
        
        generated_tokens = []
        for _ in range(126):
            outputs = decode_method.execute([current_tokens, current_positions, encodings])
            logits = outputs[0]
            next_token = torch.argmax(logits[0, -1, :]).item()
            
            if next_token == 50256: # EOS
                break
                
            generated_tokens.append(next_token)
            current_tokens = torch.tensor([[next_token]], dtype=torch.long)
            current_positions = torch.tensor([len(tokens) + len(generated_tokens) - 1], dtype=torch.long)

        hypothesis_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        elapsed = time.perf_counter() - start_time
        
        # [3] Score
        norm_ref = normalize_text(reference_text)
        norm_hyp = normalize_text(hypothesis_text)
        wer = calculate_wer(norm_ref, norm_hyp)
        
        results.append({
            "name": base,
            "wer": wer,
            "time": elapsed,
            "ref": reference_text,
            "hyp": hypothesis_text
        })
        
        wer_color = GREEN if wer < 0.1 else (YELLOW if wer < 0.3 else RED)
        print(f"{base:<12} | {wer_color}{wer:>7.2%}{RESET} | {elapsed:>9.2f} | {hypothesis_text[:50]}...")

    # Global Scores
    avg_wer = sum(r['wer'] for r in results) / len(results) if results else 0
    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    
    print("\n" + BOLD + "="*50 + RESET)
    print(BOLD + "GLOBAL EVALUATION SUMMARY" + RESET)
    print(BOLD + "="*50 + RESET)
    print(f"{CYAN}Average WER:      {BOLD}{avg_wer:.2%}{RESET}")
    print(f"{CYAN}Average Time:     {BOLD}{avg_time:.2f} s{RESET}")
    print(f"{CYAN}Samples Processed: {len(results)}{RESET}")
    print(BOLD + "="*50 + RESET + "\n")

if __name__ == "__main__":
    main()
