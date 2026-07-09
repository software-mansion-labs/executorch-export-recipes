import subprocess
import shutil
import os

def main():
    models = ["tiny", "tiny.en", "base", "base.en", "small", "small.en"]
    out_names = {
        "tiny": "whisper_tiny_coreml_fp16.pte",
        "tiny.en": "whisper_tiny_en_coreml_fp16.pte",
        "base": "whisper_base_coreml_fp16.pte",
        "base.en": "whisper_base_en_coreml_fp16.pte",
        "small": "whisper_small_coreml_fp16.pte",
        "small.en": "whisper_small_en_coreml_fp16.pte",
    }

    os.makedirs("output", exist_ok=True)
    scratch_dir = "/Users/bhanc/workspace/scratch"
    os.makedirs(scratch_dir, exist_ok=True)

    for model in models:
        out_file = os.path.join("output", out_names[model])
        print(f"\n==================================================")
        print(f"Exporting CoreML model: {model} -> {out_file}")
        print(f"==================================================")
        
        cmd = [
            ".venv/bin/python", "export_coreml.py",
            "--model-type", model,
            "--dtype", "float16",
            "--output", out_file
        ]
        subprocess.run(cmd, check=True)
        
        dst = os.path.join(scratch_dir, out_names[model])
        print(f"Copying {out_file} -> {dst}")
        shutil.copy2(out_file, dst)

    print("\n==================================================")
    print("ALL COREML MODELS RE-EXPORTED AND COPIED TO SCRATCH!")
    print("==================================================")

if __name__ == "__main__":
    main()
