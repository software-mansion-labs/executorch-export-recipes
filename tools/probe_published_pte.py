#!/usr/bin/env python3
"""Diff a published .pte against the config the harness resolves for it.

An ExecuTorch .pte is [32B header | program flatbuffer | padding | weight
segments]. The extended header at offset 8 carries the flatbuffer's size, so the
whole contract -- method IO shapes/dtypes and the LLM runner constants -- is
readable from the first few hundred KB. This tool range-requests only that
prefix, so checking a 8 GB bf16 .pte costs well under a megabyte.

Use it whenever you touch an LLM config or are about to re-publish: the config
is a claim about the artifact, and only the artifact can settle it.

    python tools/probe_published_pte.py <model> [--size S] [--backend B]
    python tools/probe_published_pte.py --url https://.../model.pte

Needs an environment with `executorch` importable (any model's uv project).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SCALAR = {0: "uint8", 1: "int8", 2: "int16", 3: "int32", 4: "int64", 5: "float16",
           6: "float32", 7: "float64", 11: "bool", 15: "bfloat16"}


def _range(url: str, start: int, end: int) -> bytes:
    r = subprocess.run(["curl", "-sL", "-H", f"Range: bytes={start}-{end}", url],
                       capture_output=True, check=True)
    return r.stdout


def _parse_flatbuffer(data: bytes):
    """Program from the flatbuffer prefix alone.

    Deliberately not `deserialize_pte_binary`: that one restores the weight
    segments, which is exactly what we avoided downloading. The private helper
    was renamed between executorch versions, so try both spellings.
    """
    import executorch.exir._serialize._program as prog_mod

    if hasattr(prog_mod, "_flatbuffer_to_program"):        # executorch >= 1.4
        return prog_mod._flatbuffer_to_program(data)
    return prog_mod._json_to_program(                       # older
        prog_mod._program_flatbuffer_to_json(data))


def read_program(data: bytes):
    """Parse an in-memory .pte prefix (header + program flatbuffer)."""
    from executorch.exir._serialize._program import _ExtendedHeader

    eh = _ExtendedHeader.from_bytes(data[8:72])
    if not eh.is_valid():
        raise SystemExit("no valid ExecuTorch extended header")
    return _parse_flatbuffer(data[:eh.program_size])


def fetch_program(url: str):
    """Return the parsed program flatbuffer, without downloading the weights."""
    from executorch.exir._serialize._program import _ExtendedHeader

    eh = _ExtendedHeader.from_bytes(_range(url, 0, 63)[8:])
    if not eh.is_valid():
        raise SystemExit(f"no valid ExecuTorch extended header at {url}")
    data = _range(url, 0, eh.program_size - 1)
    if len(data) != eh.program_size:
        raise SystemExit(f"short read: {len(data)} of {eh.program_size} bytes")
    return _parse_flatbuffer(data)


def summarize(program) -> dict:
    out = {"methods": {}, "constants": {}}
    for plan in program.execution_plan:
        vals = plan.values

        def tensor(idx):
            v = vals[idx].val
            return {"shape": list(v.sizes),
                    "dtype": _SCALAR.get(int(v.scalar_type), int(v.scalar_type))}

        if plan.name == "forward":
            out["methods"]["forward"] = {"inputs": [tensor(i) for i in plan.inputs],
                                         "outputs": [tensor(i) for i in plan.outputs]}
        else:
            got = []
            for i in plan.outputs:
                v = vals[i].val
                got.append(getattr(v, "int_val", getattr(v, "bool_val", repr(v))))
            # get_eos_ids is a list method even when it carries a single id
            out["constants"][plan.name] = (
                got if len(got) != 1 or plan.name == "get_eos_ids" else got[0])
    return out


def _dim(d):
    # A config range {min,max} and the .pte's recorded upper-bound size describe
    # the same dynamic dim; compare on the bound only.
    return str(d.get("max", "?")) if isinstance(d, dict) else str(d)


def _io(t):
    return "[" + ",".join(_dim(d) for d in t["shape"]) + "]:" + t["dtype"]


def diff(variant, got) -> list[str]:
    out = []
    cfg = {"inputs": [{"shape": t.shape, "dtype": t.dtype} for t in variant.methods["forward"].inputs],
           "outputs": [{"shape": t.shape, "dtype": t.dtype} for t in variant.methods["forward"].outputs]}
    for side in ("inputs", "outputs"):
        pte = got["methods"]["forward"][side]
        if len(cfg[side]) != len(pte):
            out.append(f"{side}: config has {len(cfg[side])}, .pte has {len(pte)}")
        for i, (c, g) in enumerate(zip(cfg[side], pte)):
            if _io(c) != _io(g):
                out.append(f"{side}[{i}]: config {_io(c)}, .pte {_io(g)}")
    for k in sorted(set(variant.constants) | set(got["constants"])):
        c, g = variant.constants.get(k, "<absent>"), got["constants"].get(k, "<absent>")
        if c != g:
            out.append(f"{k}: config {c}, .pte {g}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="model name as `exporter list` prints it")
    ap.add_argument("--size")
    ap.add_argument("--backend", default="xnnpack")
    ap.add_argument("--url", help="probe one .pte (URL or local path) and dump its contract")
    ap.add_argument("--local", help="check a local .pte against <model>'s config")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.url:
        prog = (read_program(Path(args.url).read_bytes())
                if Path(args.url).exists() else fetch_program(args.url))
        print(json.dumps(summarize(prog), indent=2))
        return 0
    if not args.model:
        ap.error("give a model name or --url")

    from exporter import registry
    spec = registry.load(args.model, size=args.size, backend=args.backend)

    if args.local:
        got = summarize(read_program(Path(args.local).read_bytes()))
        name = Path(args.local).name
        v = next((x for x in spec.variants if x.file == name), None)
        if v is None:
            ap.error(f"{name} is not a variant of {args.model}: "
                     f"{[x.file for x in spec.variants]}")
        d = diff(v, got)
        print(f"{args.local}: {'matches its config' if not d else str(len(d)) + ' difference(s)'}")
        for line in d:
            print(f"    {line}")
        return 1 if d else 0
    hf = json.loads((registry.MODELS_DIR / args.model / "model.json").read_text())["hf"]
    prefix = hf.rstrip("/") + "/resolve/main/"
    sub = f"{args.size}/{args.backend}/" if args.size else f"{args.backend}/"

    results, failed = {}, 0
    for v in spec.variants:
        got = summarize(fetch_program(prefix + sub + v.file))
        d = diff(v, got)
        results[v.file] = d
        failed += bool(d)
        if not args.json:
            print(f"{v.file}: {'matches its config' if not d else str(len(d)) + ' difference(s)'}")
            for line in d:
                print(f"    {line}")
    if args.json:
        print(json.dumps(results, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
