import argparse
import json as _json
import sys

from . import registry
from .errors import ExporterError
from .pipeline import _select, export_all, export_variant, plan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="exporter")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list").add_argument("--json", action="store_true")

    d = sub.add_parser("describe")
    d.add_argument("model")
    d.add_argument("--size", default=None)
    d.add_argument("--backend", default=None)
    d.add_argument("--json", action="store_true")

    e = sub.add_parser("export")
    e.add_argument("model")
    e.add_argument("--variant", default=None)
    e.add_argument("--all-variants", action="store_true")
    e.add_argument("--local-spec", default=None)
    e.add_argument("--size", default=None)
    e.add_argument("--backend", default=None)
    e.add_argument("--output-dir", default=None)
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--json", action="store_true")

    return p


def _emit(obj, json_mode):
    print(_json.dumps(obj) if json_mode else obj)


def _plan_dict(p):
    return {
        "model": p.model, "backend": p.backend,
        "export_script": p.export_script, "variants": p.variants,
        "output_dir": p.output_dir,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jm = getattr(args, "json", False)
    try:
        if args.command == "list":
            _emit({"models": registry.list_models()}, jm)
            return 0

        if args.command == "describe":
            spec = registry.load(args.model, size=args.size, backend=args.backend)
            script = registry.export_script_path(args.model, args.backend)
            p = plan(spec, export_script=str(script), which=None, output_dir=spec.model)
            d = _plan_dict(p)
            d["size"] = args.size
            _emit(d, jm)
            return 0

        if args.command == "export":
            spec = registry.load(args.model, local_spec=args.local_spec, size=args.size, backend=args.backend)
            script = registry.export_script_path(args.model, args.backend)
            which = "all" if args.all_variants else args.variant
            out_dir = args.output_dir or spec.model
            p = plan(spec, export_script=str(script), which=which, output_dir=out_dir)
            if args.dry_run:
                d = _plan_dict(p)
                d["size"] = args.size
                _emit(d, jm)
                return 0
            registry.check_extras(args.model)
            use_export_all = args.all_variants and registry.is_multi_pte(args.model)
            if use_export_all:
                all_results = export_all(spec, export_script=str(script), output_dir=out_dir, size=args.size)
                results = [{"file": r.file, "precision": r.precision} for r in all_results]
            else:
                results = []
                for variant in _select(spec, which):
                    r = export_variant(spec, variant, export_script=str(script), output_dir=out_dir, size=args.size)
                    results.append({"file": r.file, "precision": r.precision})
            _emit({"model": spec.model, "results": results}, jm)
            return 0

    except ExporterError as err:
        if jm:
            print(_json.dumps(err.to_dict()))
        else:
            print(f"error[{err.code}]: {err.message}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
