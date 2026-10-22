"""Dump a .pte's EFFECTIVE model spec — exactly what `model.schema` holds at runtime.

That is the MethodMeta-derived spec (every dimension `constant`) with the
`get_model_schema` JSON overlaid on top of it, method by method. Emitted as one
JSON object per .pte so the TypeScript side can be validated against real
artifacts without a device.

    uv run --project exporter_models/<model> python tools/dump_schema.py <out.json> <pte>...
"""
import json
import sys
from pathlib import Path

_DTYPE = {0: "uint8", 3: "int32", 4: "int64", 6: "float32"}


def _param(value):
    from executorch.exir.schema import Tensor

    if not isinstance(value, Tensor):
        return None
    return {
        "kind": "Tensor",
        "dtype": _DTYPE.get(int(value.scalar_type), str(value.scalar_type)),
        "shape": [{"kind": "constant", "value": int(s)} for s in value.sizes],
    }


def effective_spec(pte_path):
    from executorch.exir._serialize._program import deserialize_pte_binary

    program = deserialize_pte_binary(Path(pte_path).read_bytes()).program
    plans = {plan.name: plan for plan in program.execution_plan}

    spec = {}
    for name, plan in plans.items():
        if name.startswith("get_"):
            continue
        spec[name] = {
            "inputs": [p for p in (_param(plan.values[i].val) for i in plan.inputs) if p],
            "outputs": [p for p in (_param(plan.values[i].val) for i in plan.outputs) if p],
            "runtimeConstraints": [],
        }

    companion = plans.get("get_model_schema")
    if companion is not None:
        overlay = json.loads(companion.values[companion.outputs[0]].val.string_val)
        for name, method_spec in overlay.items():
            spec[name] = method_spec
    return spec


if __name__ == "__main__":
    out, *ptes = sys.argv[1:]
    dumped = {Path(p).name: effective_spec(p) for p in ptes}
    Path(out).write_text(json.dumps(dumped, indent=1))
    print(f"wrote {out} ({len(dumped)} artifact(s))")
