import json
import subprocess
import sys
from pathlib import Path

from .errors import EnvironmentError_
from .spec import IOTensor

DTYPE_NAMES = {
    0: "uint8", 1: "int8", 2: "int16", 3: "int32", 4: "int64",
    5: "float16", 6: "float32", 7: "float64", 8: "complex32",
    9: "complex64", 10: "complex128", 11: "bool", 15: "bfloat16",
}

_WORKER = """
import json, sys
from executorch.extension.pybindings.portable_lib import _load_for_executorch

pte_path, filter_method = sys.argv[1], sys.argv[2] or None
dtype_names = {0:"uint8",1:"int8",2:"int16",3:"int32",4:"int64",5:"float16",6:"float32",7:"float64",11:"bool",15:"bfloat16"}
def dtype(d): return dtype_names.get(int(d), f"dtype({d})")
def tensors(fn, n):
    out = []
    for i in range(n):
        try:
            t = fn(i)
            out.append({"shape": list(t.sizes()), "dtype": dtype(t.dtype())})
        except Exception:
            out.append({"shape": None, "dtype": None})
    return out
try:
    module = _load_for_executorch(pte_path)
except Exception as e:
    print(json.dumps({"error": str(e)})); sys.exit(0)
result = {}
for name in module.method_names():
    if filter_method and name != filter_method:
        continue
    meta = module.method_meta(name)
    result[name] = {
        "inputs": tensors(meta.input_tensor_meta, meta.num_inputs()),
        "outputs": tensors(meta.output_tensor_meta, meta.num_outputs()),
    }
print(json.dumps(result))
"""


def read_pte_io(pte_path, method: str | None = None) -> dict[str, dict]:
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, str(pte_path), method or ""],
        capture_output=True, text=True,
    )
    out = proc.stdout.strip()
    if not out:
        raise EnvironmentError_(
            "could not load .pte with portable runtime; is executorch installed? "
            f"stderr: {proc.stderr.strip()}"
        )
    payload = json.loads(out)
    if "error" in payload:
        raise EnvironmentError_(f"runtime load error: {payload['error']}")
    result = {}
    for name, info in payload.items():
        result[name] = {
            "inputs": [IOTensor(shape=t["shape"], dtype=t["dtype"]) for t in info["inputs"] if t["shape"] is not None],
            "outputs": [IOTensor(shape=t["shape"], dtype=t["dtype"]) for t in info["outputs"] if t["shape"] is not None],
        }
    return result
