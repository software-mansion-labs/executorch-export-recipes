"""Gemma4 export CLI.

    PYTHONPATH=. python -m gemma4.export --recipe vulkan-mm --dry-run
    PYTHONPATH=. python -m gemma4.export --recipe xnnpack-mm --out build/g4.pte

Start with --dry-run. It resolves the recipe, applies your overrides, runs every
validation rule and prints the result without building anything.
"""
# ---------------------------------------------------------------------------
# IMPORT ORDER IS LOAD-BEARING. `executorch.runtime.Runtime` must be imported
# before `executorch.kernels.quantized`, or the quantized kernel registration
# silently no-ops and `embedding_{2,4}bit.out` fails to resolve later, at
# load_method time, with an error that points nowhere near here.
# Keep these three first, above everything including `torch`.
# ---------------------------------------------------------------------------
#
# NOTE: there is deliberately no `from __future__ import annotations` here.
# That statement must be the first in the file, which would push the kernel
# imports below it. Python 3.10+ evaluates `X | None` natively, so the future
# import buys nothing and the ordering constraint wins.
from executorch.runtime import Runtime  # noqa: F401  (side effect: kernel registry)
import executorch.kernels.quantized  # noqa: F401
import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List, Optional  # noqa: E402

from . import presets, validate  # noqa: E402
from .config import Recipe  # noqa: E402

DEFAULT_CHECKPOINT = "gemma_export/gemma4_e2b"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gemma4.export",
        description="Export Gemma 4 E2B to an ExecuTorch PTE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Recipes:\n"
            "  xnnpack-text   XNNPACK, text only, single `forward` method\n"
            "  xnnpack-mm     XNNPACK, text + vision + audio, 4 methods\n"
            "  vulkan-text    Vulkan, text only, single `forward` method\n"
            "  vulkan-mm      Vulkan (audio on XNNPACK), 4 methods - UNVALIDATED\n"
            "\n"
            "See gemma4/docs/RECIPES.md for what each one is and how far to\n"
            "trust it, and gemma4/docs/LEVERS.md for what not to re-try.\n"
        ),
    )
    p.add_argument("--recipe", choices=sorted(presets.PRESETS), help="which recipe to export")
    p.add_argument("--list", action="store_true", help="print all recipes and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve, validate and print; build nothing")
    p.add_argument("--force", action="store_true",
                   help="export despite validation errors (rarely correct)")

    io = p.add_argument_group("paths")
    io.add_argument("--checkpoint", default=os.environ.get("GEMMA4_CHECKPOINT", DEFAULT_CHECKPOINT),
                    help=f"HF checkpoint directory (default: $GEMMA4_CHECKPOINT or {DEFAULT_CHECKPOINT})")
    io.add_argument("--out", help="output .pte path")

    m = p.add_argument_group("model")
    m.add_argument("--num-layers", type=int,
                   help="truncate to N layers for a smoke test (default: recipe value, 35)")
    m.add_argument("--num-kv-shared-layers", type=int)

    s = p.add_argument_group("sequence lengths")
    s.add_argument("--max-seq-len", type=int, help="KV cache depth / context limit")
    s.add_argument("--prefill-len", type=int,
                   help="token_embedding / forward seq bound")
    s.add_argument("--text-decoder-prefill-len", type=int,
                   help="text_decoder seq bound; this is what sizes the arena")

    q = p.add_argument_group("quantisation")
    q.add_argument("--group-size", type=int, help="linear quant group size")
    q.add_argument("--emb-group-size", type=int, help="main embedding group size")
    q.add_argument("--ple-bits", type=int, choices=(2, 4), help="PLE embedding bit width")
    q.add_argument("--ple-group-size", type=int)
    q.add_argument("--no-quantize-embeddings", action="store_true",
                   help="leave the main embedding table unquantized")

    a = p.add_argument_group("attention flags (override the recipe)")
    a.add_argument("--use-prebroadcast-rope", action="store_true",
                   help="Lever F: measured -32 host ops at 5L, never validated at 35L")
    a.add_argument("--no-rsqrt", action="store_true",
                   help="disable the RMSNorm rsqrt swap (Lever A')")
    return p


def _apply_overrides(recipe: Recipe, args: argparse.Namespace) -> Recipe:
    """Apply CLI overrides on top of a preset, section by section."""
    lengths = {}
    if args.num_layers is not None:
        lengths["num_layers"] = args.num_layers
    if args.num_kv_shared_layers is not None:
        lengths["num_kv_shared_layers"] = args.num_kv_shared_layers
    elif args.num_layers is not None:
        # Truncating layers without adjusting kv-sharing would leave
        # num_kv_shared_layers >= num_layers, which is nonsense. Scale it the
        # way truncated.py does: keep the same number of trailing shared layers
        # where possible, else clamp to zero.
        shared = recipe.lengths.num_kv_shared_layers
        lengths["num_kv_shared_layers"] = max(0, args.num_layers - (recipe.lengths.num_layers - shared))
    if args.max_seq_len is not None:
        lengths["max_seq_len"] = args.max_seq_len
    if args.prefill_len is not None:
        lengths["prefill_len"] = args.prefill_len
    if args.text_decoder_prefill_len is not None:
        lengths["text_decoder_prefill_len"] = args.text_decoder_prefill_len
    if lengths:
        recipe = recipe.replace_nested("lengths", **lengths)

    quant = {}
    if args.group_size is not None:
        quant["linear_group_size"] = args.group_size
    if args.emb_group_size is not None:
        quant["embedding_group_size"] = args.emb_group_size
    if args.ple_bits is not None:
        quant["ple_embedding_bits"] = args.ple_bits
    if args.ple_group_size is not None:
        quant["ple_embedding_group_size"] = args.ple_group_size
    if args.no_quantize_embeddings:
        quant["quantize_main_embeddings"] = False
    if quant:
        recipe = recipe.replace_nested("quant", **quant)

    attn = {}
    if args.use_prebroadcast_rope:
        attn["use_prebroadcast_rope"] = True
    if args.no_rsqrt:
        attn["use_rsqrt"] = False
    if attn:
        recipe = recipe.replace_nested("attn", **attn)

    # Any override means this is no longer the preset that was validated.
    if lengths or quant or attn:
        recipe = recipe.overridden(
            validated_on_device=False,
            notes=(recipe.notes + " " if recipe.notes else "")
            + "Modified from the preset by CLI overrides; the preset's "
              "on-device validation no longer applies.",
        )
    return recipe


def _print_list() -> None:
    print("Available recipes:\n")
    for name, r in presets.PRESETS.items():
        mark = "ok " if r.validated_on_device else "!! "
        print(f"  {mark}{name:<14} {r.backend:<8} {r.modality:<11} <- {r.source_script}")
    print("\n  !! = never validated on device; read docs/RECIPES.md")


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        _print_list()
        return 0

    if not args.recipe:
        print("error: --recipe is required (or use --list)", file=sys.stderr)
        return 2

    recipe = _apply_overrides(presets.get(args.recipe), args)

    print(recipe.summary())
    print()

    errors, warnings = validate.check(recipe)
    for w in warnings:
        print(f"  warning  {w}")
    for e in errors:
        print(f"  ERROR    {e}", file=sys.stderr)
    if warnings or errors:
        print()

    if errors and not args.force:
        print(
            f"refusing to export: {len(errors)} validation error(s). "
            "Fix them, or pass --force if you are certain.",
            file=sys.stderr,
        )
        return 1

    print("constant_methods:")
    for k, v in recipe.constant_methods().items():
        print(f"  {k:<26} {v}")
    print()

    if args.dry_run:
        print("--dry-run: stopping before build.")
        return 0

    if not args.out:
        print("error: --out is required unless --dry-run", file=sys.stderr)
        return 2

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(
            f"error: checkpoint directory not found: {checkpoint}\n"
            "Pass --checkpoint or set GEMMA4_CHECKPOINT.",
            file=sys.stderr,
        )
        return 2

    # Imported here, not at module scope: it pulls in torch, torchao and the
    # model, which makes --list and --dry-run needlessly slow.
    from . import pipeline

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pipeline.export(recipe, checkpoint=checkpoint, out_path=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
