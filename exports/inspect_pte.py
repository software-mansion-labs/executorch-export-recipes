#!/usr/bin/env python3
"""
Inspect a .pte file: backends, methods, input/output shapes and dtypes.

Usage:
    python3 scripts/inspect_pte.py path/to/model.pte
    python3 scripts/inspect_pte.py path/to/model.pte forward_384
"""

import json
import os
import re
import subprocess
import sys


KNOWN_BACKENDS = [
    "XnnpackBackend", "CoreMLBackend", "MPSBackend",
    "QnnBackend", "VulkanBackend", "HtpBackend",
]

DTYPE_NAMES = {
    0: "uint8", 1: "int8", 2: "int16", 3: "int32", 4: "int64",
    5: "float16", 6: "float32", 7: "float64", 8: "complex32",
    9: "complex64", 10: "complex128", 11: "bool", 15: "bfloat16",
}

# Worker script run in a subprocess so crashes don't kill the main process
WORKER = """
import json, sys
from executorch.extension.pybindings.portable_lib import _load_for_executorch

pte_path, filter_method = sys.argv[1], sys.argv[2] or None
try:
    module = _load_for_executorch(pte_path)
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(0)

dtype_names = {0:"uint8",1:"int8",2:"int16",3:"int32",4:"int64",5:"float16",6:"float32",7:"float64",11:"bool",15:"bfloat16"}
def dtype(d): return dtype_names.get(int(d), f"dtype({d})")
def tensors(fn, n):
    out = []
    for i in range(n):
        try:
            t = fn(i); out.append({"shape": list(t.sizes()), "dtype": dtype(t.dtype())})
        except: out.append(None)
    return out

result = {}
for name in module.method_names():
    if filter_method and name != filter_method:
        continue
    try:
        meta = module.method_meta(name)
        result[name] = {
            "inputs": tensors(meta.input_tensor_meta, meta.num_inputs()),
            "outputs": tensors(meta.output_tensor_meta, meta.num_outputs()),
        }
    except Exception as e:
        result[name] = {"error": str(e)}
print(json.dumps(result))
"""


def detect_backends(data):
    found = [b for b in KNOWN_BACKENDS if b.encode() in data]
    return found if found else ["(none / CPU only)"]


def scan_method_names(data):
    names = set()
    for m in re.finditer(rb'(forward(?:_\d+)?)', data):
        names.add(m.group(1).decode())
    return sorted(names)


def load_meta(pte_path, filter_method):
    result = subprocess.run(
        [sys.executable, "-c", WORKER, pte_path, filter_method or ""],
        capture_output=True, text=True
    )
    stdout = result.stdout.strip()
    if not stdout:
        return None, result.stderr
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError:
        return None, stdout + result.stderr


def fmt_tensor(t):
    if t is None:
        return "?"
    return f"{t['shape']} {t['dtype']}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pte_path = sys.argv[1]
    filter_method = sys.argv[2] if len(sys.argv) > 2 else None

    with open(pte_path, "rb") as f:
        data = f.read()

    backends = detect_backends(data)
    print(f"\nFile:     {os.path.basename(pte_path)}")
    print(f"Size:     {len(data) / 1024 / 1024:.1f} MB")
    print(f"Backends: {', '.join(backends)}")
    print()

    meta, err = load_meta(pte_path, filter_method)

    if meta is None:
        print("Could not load method metadata with portable runtime.")
        print("(Hardware backend models — Vulkan/QNN/CoreML — cannot run on host.)")
        print("\nDetected method names (from binary scan):")
        for name in scan_method_names(data):
            if not filter_method or name == filter_method:
                print(f"  {name}")
        return

    if "error" in meta:
        print(f"Load error: {meta['error']}")
        print("\nDetected method names (from binary scan):")
        for name in scan_method_names(data):
            if not filter_method or name == filter_method:
                print(f"  {name}")
        return

    for name, info in meta.items():
        print(f"Method: {name}")
        if "error" in info:
            print(f"  (error: {info['error']})")
        else:
            for i, t in enumerate(info["inputs"]):
                print(f"  in  [{i}]: {fmt_tensor(t)}")
            for i, t in enumerate(info["outputs"]):
                print(f"  out [{i}]: {fmt_tensor(t)}")
        print()


if __name__ == "__main__":
    main()
