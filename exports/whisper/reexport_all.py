import subprocess
import shutil
import os

def main():
    models = ["small", "small.en"]
    out_names = {
        "tiny": "whisper_tiny_xnnpack_fp32.pte",
        "tiny.en": "whisper_tiny_en_xnnpack_fp32.pte",
        "base": "whisper_base_xnnpack_fp32.pte",
        "base.en": "whisper_base_en_xnnpack_fp32.pte",
        "small": "whisper_small_xnnpack_fp32.pte",
        "small.en": "whisper_small_en_xnnpack_fp32.pte",
    }

    os.makedirs("output", exist_ok=True)
    scratch_dir = "/Users/bhanc/workspace/scratch"

    for model in models:
        out_file = os.path.join("output", out_names[model])
        print(f"\n==================================================")
        print(f"Exporting model: {model} -> {out_file}")
        print(f"==================================================")
        
        cmd = [
            "python", "export_xnnpack.py",
            "--model-type", model,
            "--dtype", "float32",
            "--output", out_file
        ]
        subprocess.run(cmd, check=True)
        
        dst = os.path.join(scratch_dir, out_names[model])
        print(f"Copying {out_file} -> {dst}")
        shutil.copy2(out_file, dst)

    print("\n==================================================")
    print("ALL MODELS RE-EXPORTED AND COPIED TO SCRATCH!")
    print("==================================================")

if __name__ == "__main__":
    main()
