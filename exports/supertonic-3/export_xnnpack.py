#!/usr/bin/env python3
"""Export Supertonic 3 (PyTorch port) to a bundled ExecuTorch XNNPACK .pte.

The four sub-models are lowered as four methods in a single ``.pte`` — mirroring
the ONNX layout — so the host drives the same pipeline (text preprocessing,
noise sampling, the flow-matching loop) around them:

    duration_predictor(text_ids, style_dp,  text_mask)               -> duration
    text_encoder      (text_ids, style_ttl, text_mask)               -> text_emb
    vector_estimator  (noisy_latent, text_emb, style_ttl, text_mask,
                       latent_mask, current_step, total_step)         -> denoised_latent
    vocoder           (latent)                                        -> wav

Dynamic shapes:
  * text length T is dynamic (VITS relative attention needs the size-oblivious
    export mode; see supertonic_torch.modules)
  * latent length L is dynamic

Usage:
    python scripts/export_xnnpack.py --output supertonic3_xnnpack.pte
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.fx.experimental._config as fx_config
from torch.export import Dim

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

from supertonic_torch.convert import load_models

ROOT = _ROOT

N_STYLE = 50
STYLE_DIM = 256
DP_STYLE = (8, 16)
TEXT_EMB_DIM = 256
LATENT_DIM = 144  # ldim(24) * chunk_compress_factor(6)


@contextlib.contextmanager
def size_oblivious_export():
    """Treat backed sizes size-obliviously so the VITS relative-attention
    reshapes export with a dynamic sequence length (their divisibility
    identities are always true but torch.export cannot prove them)."""
    prev = fx_config.backed_size_oblivious
    fx_config.backed_size_oblivious = True
    try:
        yield
    finally:
        fx_config.backed_size_oblivious = prev


def build_example_inputs(t_text=64, t_latent=80):
    text_ids = torch.randint(1, 8000, (1, t_text), dtype=torch.long)
    text_mask = torch.ones(1, 1, t_text)
    style_ttl = torch.randn(1, N_STYLE, STYLE_DIM)
    style_dp = torch.randn(1, *DP_STYLE)

    noisy_latent = torch.randn(1, LATENT_DIM, t_latent)
    text_emb = torch.randn(1, TEXT_EMB_DIM, t_text)
    latent_mask = torch.ones(1, 1, t_latent)
    current_step = torch.tensor([2.0])
    total_step = torch.tensor([8.0])

    return dict(
        text_ids=text_ids, text_mask=text_mask, style_ttl=style_ttl,
        style_dp=style_dp, noisy_latent=noisy_latent, text_emb=text_emb,
        latent_mask=latent_mask, current_step=current_step, total_step=total_step,
    )


def export_methods(models, max_text=512, max_latent=1000):
    ex = build_example_inputs()

    # Each method gets its own fresh Dim objects.
    with torch.inference_mode():
        print("  - exporting duration_predictor (dynamic T)...")
        T = Dim("T", min=8, max=max_text)
        dp = torch.export.export(
            models["duration_predictor"],
            (ex["text_ids"], ex["style_dp"], ex["text_mask"]),
            dynamic_shapes=({1: T}, None, {2: T}),
            strict=False,
        )

        print("  - exporting text_encoder (dynamic T)...")
        T = Dim("T", min=8, max=max_text)
        te = torch.export.export(
            models["text_encoder"],
            (ex["text_ids"], ex["style_ttl"], ex["text_mask"]),
            dynamic_shapes=({1: T}, None, {2: T}),
            strict=False,
        )

        print("  - exporting vector_estimator (dynamic T, L)...")
        T = Dim("T", min=8, max=max_text)
        L = Dim("L", min=4, max=max_latent)
        ve = torch.export.export(
            models["vector_estimator"],
            (ex["noisy_latent"], ex["text_emb"], ex["style_ttl"], ex["text_mask"],
             ex["latent_mask"], ex["current_step"], ex["total_step"]),
            dynamic_shapes=({2: L}, {2: T}, None, {2: T}, {2: L}, None, None),
            strict=False,
        )

        print("  - exporting vocoder (dynamic L)...")
        L = Dim("L", min=4, max=max_latent)
        vo = torch.export.export(
            models["vocoder"],
            (ex["noisy_latent"],),
            dynamic_shapes=({2: L},),
            strict=False,
        )

    return {
        "duration_predictor": dp,
        "text_encoder": te,
        "vector_estimator": ve,
        "vocoder": vo,
    }


def lower_method(name, exported):
    """Lower a single exported method to an ExecuTorch XNNPACK program."""
    edge = to_edge_transform_and_lower(
        {"forward": exported},
        partitioner={"forward": [XnnpackPartitioner()]},
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
        transform_passes=[AddmmToLinearTransform()],
    )
    try:
        from executorch.devtools.backend_debug.delegation_info import (
            get_delegation_info,
        )

        info = get_delegation_info(edge.exported_program("forward").graph_module)
        total = info.num_delegated_nodes + info.num_non_delegated_nodes
        print(f"    DELEGATION {name}: {info.num_delegated_nodes}/{total} nodes")
    except Exception as e:  # delegation stats are best-effort
        print(f"    (delegation stats unavailable: {e})")
    return edge.to_executorch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "supertonic3.pt"))
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "xnnpack"),
                        help="Directory to write the four .pte files into")
    parser.add_argument("--max-text", type=int, default=512, help="Max text length")
    # Rotary position buffer (vector_estimator.increments) is sized 1000.
    parser.add_argument("--max-latent", type=int, default=1000, help="Max latent frames")
    args = parser.parse_args()

    print(f"[1/3] Loading PyTorch models from {args.checkpoint}...")
    models = load_models(checkpoint_path=args.checkpoint)

    # Size-oblivious mode must stay active through BOTH export and lowering:
    # the memory planner needs it to bound the text encoder's dynamic quadratic
    # attention activation, otherwise it reserves a phantom >512 MB buffer that
    # overflows the runtime arena at load time.
    with size_oblivious_export():
        print("[2/3] Exporting the four methods...")
        methods = export_methods(models, args.max_text, args.max_latent)

        # One .pte per method: the four sub-models total ~400 MB, which
        # overflows ExecuTorch's single constant arena if bundled into one file.
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[3/3] Lowering each method to ExecuTorch XNNPACK -> {out_dir}/")
        t0 = time.time()
        total_mb = 0.0
        for name, exported in methods.items():
            program = lower_method(name, exported)
            path = out_dir / f"{name}.pte"
            with open(path, "wb") as f:
                program.write_to_file(f)
            mb = path.stat().st_size / 1e6
            total_mb += mb
            print(f"    wrote {path} ({mb:.1f} MB)")
    print(f"    lowering completed in {time.time() - t0:.1f}s")
    print(f"Success! Four XNNPACK .pte files ({total_mb:.1f} MB total) in '{out_dir}'.")


if __name__ == "__main__":
    main()
