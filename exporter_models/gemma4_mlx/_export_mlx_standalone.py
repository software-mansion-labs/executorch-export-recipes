#!/usr/bin/env python3
"""Standalone: export Gemma 4 E2B to the ExecuTorch MLX (Apple GPU) backend.

Self-contained — copy this single file into any project with ExecuTorch
installed from source.

  python export_mlx_standalone.py --model gemma4-e2b      --checkpoint <hf_snapshot_dir>
  python export_mlx_standalone.py --model gemma4-e2b-omni --checkpoint <hf_snapshot_dir>

`--checkpoint` is a HuggingFace snapshot dir containing model.safetensors
(auto-detected from the HF cache if omitted). Find it with:
  ls ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*/model.safetensors

Settings baked in (on-device-tested): int4 / group_size 64, bf16,
compile-friendly text decoder (no .item() → enables the mx::compile fast path),
2048 context with chunked prefill.

Runtime note: the compile-friendly text model REQUIRES the matching MLX runtime
(rope_t custom op, IndexCopy cache, compiled execute path) in the libbackend_mlx
you link into your app.

Prerequisite: ExecuTorch built from source, plus `torch`.
"""
import argparse
import glob
import os
import sys
import tempfile

GROUP_SIZE = 64
CONTEXT_LEN = 2048
MODELS = ("gemma4-e2b", "gemma4-e2b-omni")


def resolve_checkpoint(arg):
    if arg and os.path.exists(arg):
        return os.path.abspath(arg)
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/"
            "snapshots/*/model.safetensors"
        )
    )
    if not hits:
        raise SystemExit(
            "Gemma needs --checkpoint <hf_snapshot_dir> (a dir with "
            "model.safetensors). None found in the HF cache."
        )
    return os.path.dirname(hits[0])


def export(model_key: str, out_path: str, checkpoint: str) -> None:
    if model_key == "gemma4-e2b":
        from executorch.examples.models.gemma4.export_gemma4_mlx import (
            export_gemma4_e2b_mlx,
        )

        export_gemma4_e2b_mlx(
            checkpoint_path=checkpoint,
            output_path=out_path,
            max_seq_len=CONTEXT_LEN,
            qlinear="4w",
            qembedding="4w",
            group_size=GROUP_SIZE,
            dtype="bf16",
            compile_friendly=True,
        )
    else:  # gemma4-e2b-omni (text + vision + audio)
        from executorch.examples.models.gemma4 import (
            export_gemma4_omni_mlx_v3 as omni,
        )

        sys.argv = [
            "export_gemma4_omni_mlx_v3.py",
            "--checkpoint_path", checkpoint,
            "--group_size", str(GROUP_SIZE),
            "--output", out_path,
        ]
        omni.main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--out", default="mlx_exports")
    ap.add_argument("--checkpoint", help="HF snapshot dir with model.safetensors.")
    args = ap.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.abspath(
        os.path.join(args.out, args.model.replace(".", "_") + "_mlx.pte")
    )
    # Neutral cwd so ExecuTorch's cwd-based import-prefix heuristic resolves the
    # installed `executorch.examples.*` package regardless of invocation dir.
    os.chdir(tempfile.gettempdir())
    export(args.model, out, checkpoint)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
