import argparse
import json as _json
import sys
from pathlib import Path

from . import registry
from .errors import ExporterError
from .pipeline import _select, compare_io, export_all, export_variant, plan
from .pte_inspect import read_pte_io
from .spec import load_spec_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="exporter")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list").add_argument("--json", action="store_true")

    d = sub.add_parser("describe")
    d.add_argument("model")
    d.add_argument("--size", default=None)
    d.add_argument("--json", action="store_true")

    e = sub.add_parser("export")
    e.add_argument("model")
    e.add_argument("--variant", default=None)
    e.add_argument("--all-variants", action="store_true")
    e.add_argument("--local-spec", default=None)
    e.add_argument("--size", default=None)
    e.add_argument("--atol", type=float, default=None)
    e.add_argument("--output-dir", default=None)
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--json", action="store_true")

    v = sub.add_parser("verify")
    v.add_argument("pte")
    v.add_argument("config")
    v.add_argument("--json", action="store_true")

    return p


def _emit(obj, json_mode):
    print(_json.dumps(obj) if json_mode else obj)


def _plan_dict(p):
    return {
        "model": p.model, "backend": p.backend,
        "export_script": p.export_script, "variants": p.variants,
        "output_dir": p.output_dir, "atol": p.atol,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jm = getattr(args, "json", False)
    try:
        if args.command == "list":
            _emit({"models": registry.list_models()}, jm)
            return 0

        if args.command == "describe":
            spec = registry.load(args.model, size=args.size)
            script = registry.export_script_path(args.model)
            p = plan(spec, export_script=str(script), which=None, output_dir=spec.model, atol=None)
            d = _plan_dict(p)
            d["size"] = args.size
            _emit(d, jm)
            return 0

        if args.command == "export":
            spec = registry.load(args.model, local_spec=args.local_spec, size=args.size)
            script = registry.export_script_path(args.model)
            which = "all" if args.all_variants else args.variant
            out_dir = args.output_dir or spec.model
            p = plan(spec, export_script=str(script), which=which, output_dir=out_dir, atol=args.atol)
            if args.dry_run:
                d = _plan_dict(p)
                d["size"] = args.size
                _emit(d, jm)
                return 0
            use_export_all = args.all_variants and len({v.file for v in spec.variants}) > 1
            if use_export_all:
                all_results = export_all(spec, export_script=str(script), output_dir=out_dir, atol=p.atol, size=args.size)
                results = [
                    {"file": r.file, "precision": r.precision,
                     "io_ok": r.io_ok, "parity_ok": r.parity_ok, "bench_ms": r.bench_ms}
                    for r in all_results
                ]
            else:
                results = []
                for variant in _select(spec, which):
                    r = export_variant(spec, variant, export_script=str(script), output_dir=out_dir, atol=p.atol, size=args.size)
                    results.append({
                        "file": r.file, "precision": r.precision,
                        "io_ok": r.io_ok, "parity_ok": r.parity_ok, "bench_ms": r.bench_ms,
                    })
            _emit({"model": spec.model, "results": results}, jm)
            return 0

        if args.command == "verify":
            spec = load_spec_file(args.config)
            pte_name = Path(args.pte).name
            variant = next((v for v in spec.variants if v.file == pte_name), spec.default_variant())
            actual = read_pte_io(args.pte)
            check = compare_io(variant.methods, actual)
            _emit({"ok": check.ok, "deltas": check.deltas}, jm)
            return 0 if check.ok else 1

    except ExporterError as err:
        if jm:
            print(_json.dumps(err.to_dict()))
        else:
            print(f"error[{err.code}]: {err.message}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
