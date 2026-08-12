"""Verify a .pte against the RNE model-spec contract (src/core/schema.ts + cpp/core/schema.cpp).

Reads the `get_model_schema` constant method (if present), validates the JSON
against the repo-root model.schema.json, then cross-checks it against the
program's compiled method metadata the same way the native loader does:

  - every method named in the JSON exists in the program
  - input/output counts and tensor dtypes match
  - ranks match, `constant` dims equal the compiled dim
  - `range` max / `enum` choices never exceed the compiled allocation bound

Methods absent from the JSON are reported as MethodMeta fallback (all-constant),
which is what the runtime derives for them.

Works straight off the flatbuffer — the ET pybindings do not load against the
nightly torch these envs pin, so nothing here executes the program.

    uv run --project exporter_models/<model> python tools/verify_schema.py <pte>...
"""
import json
import sys
from pathlib import Path

import jsonschema

_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = json.loads((_ROOT / "model.schema.json").read_text())

# executorch.exir.schema.ScalarType -> the DType names the RNE schema uses.
_DTYPE = {0: "uint8", 3: "int32", 4: "int64", 6: "float32"}


def _dim_values(dim):
    """(kind, [extreme values the dim may take]) for one ConcreteDim entry."""
    kind = dim["kind"]
    if kind == "constant":
        return kind, [dim["value"]]
    if kind == "range":
        r = dim["range"]
        return kind, [r["min"], r["max"]]
    return kind, list(dim["choices"])


def _tensors(plan, indices):
    """The Tensor EValues behind a plan's input/output index list, in order."""
    from executorch.exir.schema import Tensor

    out = []
    for i in indices:
        val = plan.values[i].val
        if isinstance(val, Tensor):
            out.append(val)
    return out


def _check_params(ctx, specs, tensors, errors):
    for i, (spec, tensor) in enumerate(zip(specs, tensors)):
        where = f"{ctx} #{i}"
        got = _DTYPE.get(int(tensor.scalar_type), str(tensor.scalar_type))
        if spec["dtype"] != got:
            errors.append(f"{where}: dtype {spec['dtype']} != compiled {got}")
        sizes = list(tensor.sizes)
        if len(spec["shape"]) != len(sizes):
            errors.append(f"{where}: rank {len(spec['shape'])} != compiled {len(sizes)}")
            continue
        for d, dim in enumerate(spec["shape"]):
            kind, values = _dim_values(dim)
            bound = sizes[d]
            if kind == "constant":
                if values[0] != bound:
                    errors.append(f"{where} dim {d}: constant {values[0]} != compiled {bound}")
            elif max(values) > bound:
                errors.append(f"{where} dim {d}: {kind} max {max(values)} > compiled bound {bound}")


def verify(pte_path):
    from executorch.exir._serialize._program import deserialize_pte_binary

    print(f"== {pte_path}")
    program = deserialize_pte_binary(Path(pte_path).read_bytes()).program
    plans = {plan.name: plan for plan in program.execution_plan}

    if "get_model_schema" not in plans:
        print("   no get_model_schema -> full MethodMeta fallback (all dims constant)")
        return True

    plan = plans["get_model_schema"]
    spec = json.loads(plan.values[plan.outputs[0]].val.string_val)
    jsonschema.validate(spec, _SCHEMA)
    print(f"   get_model_schema OK against model.schema.json ({len(spec)} method(s))")

    errors = []
    for method, method_spec in sorted(spec.items()):
        if method not in plans:
            errors.append(f"method '{method}' declared in schema but not exported")
            continue
        plan = plans[method]
        ins = [p for p in method_spec["inputs"] if p["kind"] == "Tensor"]
        outs = [p for p in method_spec["outputs"] if p["kind"] == "Tensor"]
        if len(method_spec["inputs"]) != len(plan.inputs):
            errors.append(f"'{method}': {len(method_spec['inputs'])} inputs != compiled {len(plan.inputs)}")
        if len(method_spec["outputs"]) != len(plan.outputs):
            errors.append(f"'{method}': {len(method_spec['outputs'])} outputs != compiled {len(plan.outputs)}")
        _check_params(f"'{method}' input", ins, _tensors(plan, plan.inputs), errors)
        _check_params(f"'{method}' output", outs, _tensors(plan, plan.outputs), errors)
        print(f"   {method}: {len(ins)} in / {len(outs)} out, "
              f"{len(method_spec['runtimeConstraints'])} constraint(s)")

    fallback = [n for n in plans if n not in spec and not n.startswith("get_")]
    if fallback:
        print(f"   MethodMeta fallback (all-constant): {', '.join(sorted(fallback))}")

    for e in errors:
        print(f"   FAIL {e}")
    return not errors


if __name__ == "__main__":
    ok = all([verify(p) for p in sys.argv[1:]])
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
