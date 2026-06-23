import argparse
import json as _json
import sys

from . import registry
from .errors import ExporterError
from .pipeline import compare_io, export_variant, plan
from .pte_inspect import read_pte_io
from .scaffold import draft_spec
from .spec import load_spec_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="exporter")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list").add_argument("--json", action="store_true")

    d = sub.add_parser("describe")
    d.add_argument("model")
    d.add_argument("--json", action="store_true")

    e = sub.add_parser("export")
    e.add_argument("model")
    e.add_argument("--variant", default=None)
    e.add_argument("--all-variants", action="store_true")
    e.add_argument("--local-spec", default=None)
    e.add_argument("--atol", type=float, default=None)
    e.add_argument("--output-dir", default=None)
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--json", action="store_true")

    v = sub.add_parser("verify")
    v.add_argument("pte")
    v.add_argument("config")
    v.add_argument("--json", action="store_true")

    i = sub.add_parser("init")
    i.add_argument("source")
    i.add_argument("-o", "--output", default=None)
    i.add_argument("--max-seq-len", type=int, default=254)
    i.add_argument("--json", action="store_true")

    return p


def _emit(obj, json_mode):
    if json_mode:
        print(_json.dumps(obj))
    else:
        print(obj)


def _plan_dict(p):
    return {
        "model": p.model, "source": p.source, "backend": p.backend,
        "wrapper": p.wrapper, "wrapper_reason": p.wrapper_reason,
        "wrapper_kwargs": p.wrapper_kwargs, "variants": p.variants,
        "output_dir": p.output_dir, "atol": p.atol,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jm = args.json
    try:
        if args.command == "list":
            _emit({"models": registry.list_models()}, jm)
            return 0

        if args.command == "describe":
            spec = registry.load(args.model)
            p = plan(spec, which=None, output_dir=spec.model, atol=None)
            _emit(_plan_dict(p), jm)
            return 0

        if args.command == "export":
            spec = registry.load(args.model, local_spec=args.local_spec)
            which = "all" if args.all_variants else args.variant
            out_dir = args.output_dir or spec.model
            p = plan(spec, which=which, output_dir=out_dir, atol=args.atol)
            if args.dry_run:
                _emit(_plan_dict(p), jm)
                return 0
            from .pipeline import _select
            results = []
            for variant in _select(spec, which):
                r = export_variant(spec, variant, output_dir=out_dir, atol=p.atol)
                results.append({
                    "file": r.file, "precision": r.precision,
                    "io_ok": r.io_ok, "parity_max_diff": r.parity_max_diff,
                    "parity_ok": r.parity_ok, "bench_ms": r.bench_ms,
                })
            _emit({"model": spec.model, "results": results}, jm)
            return 0

        if args.command == "verify":
            from pathlib import Path
            spec = load_spec_file(args.config)
            pte_name = Path(args.pte).name
            variant = next((v for v in spec.variants if v.file == pte_name), spec.default_variant())
            actual = read_pte_io(args.pte)
            check = compare_io(variant.methods, actual)
            _emit({"ok": check.ok, "deltas": check.deltas}, jm)
            return 0 if check.ok else 1

        if args.command == "init":
            spec = draft_spec(args.source, max_seq_len=args.max_seq_len)
            text = _json.dumps(spec, indent=2)
            if args.output:
                with open(args.output, "w") as f:
                    f.write(text + "\n")
                _emit({"written": args.output}, jm)
            else:
                print(text)
            return 0

    except ExporterError as err:
        if jm:
            print(_json.dumps(err.to_dict()))
        else:
            print(f"error[{err.code}]: {err.message}", file=sys.stderr)
        return 1

    return 2
