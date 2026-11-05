"""Run a real .pte, check its contract, compare reference logits, and generate."""

import argparse
import gc
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--context-length", type=int, default=8192)
parser.add_argument("--native-runtime", type=Path)
parser.add_argument("--kernel-library", type=Path, action="append", default=[])
args = parser.parse_args()

# A matching source-built runtime is required for the fork's MLX wire format.
# Load it before importing the wheel's Python wrapper.
if args.native_runtime:
    import sys

    name = "executorch.extension.pybindings._portable_lib"
    spec = importlib.util.spec_from_file_location(name, args.native_runtime)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

import ctypes
import executorch.extension.pybindings._portable_lib as portable
from executorch.extension.pybindings.portable_lib import _load_for_executorch

ctypes.CDLL(portable.__file__, mode=ctypes.RTLD_GLOBAL)
for library in args.kernel_library:
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)

from executorch.exir._serialize._program import _ExtendedHeader, _flatbuffer_to_program

torch.set_num_threads(4)
pte = args.model
folder = pte.parent
with pte.open("rb") as f:
    header = f.read(72)
    size = _ExtendedHeader.from_bytes(header[8:72]).program_size
    f.seek(0)
    program = _flatbuffer_to_program(f.read(size))
plan = next(p for p in program.execution_plan if p.name == "forward")
report = {
    "file": str(pte),
    "bytes": pte.stat().st_size,
    "context_length": args.context_length,
    "delegates": dict(Counter(d.id for d in plan.delegates)),
    "portable_operators": [o.name + "." + o.overload for o in plan.operators],
}
schema_plan = next(p for p in program.execution_plan if p.name == "get_model_schema")
schema = json.loads(schema_plan.values[schema_plan.outputs[0]].val.string_val)
assert schema["forward"]["inputs"][0]["shape"][1]["range"] == {
    "min": 1,
    "max": 128,
    "step": 1,
}
assert schema["forward"]["outputs"][0]["dtype"] == "float32"
for side, indices in [("inputs", plan.inputs), ("outputs", plan.outputs)]:
    for expected, idx in zip(schema["forward"][side], indices):
        tensor = plan.values[idx].val
        for dim, bound in zip(expected["shape"], tensor.sizes):
            assert (
                dim["value"] if dim["kind"] == "constant" else dim["range"]["max"]
            ) == bound
print("CONTRACT", json.dumps(report), flush=True)
del program, plan
gc.collect()
reference = torch.load(args.reference, weights_only=True)
tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
model = _load_for_executorch(str(pte))
print("METHODS", model.method_names(), flush=True)
for name, expected in [
    ("get_vocab_size", 128000),
    ("get_n_layers", 30),
    ("get_max_context_len", args.context_length),
]:
    assert model.run_method(name, ())[0] == expected


def forward(ids, pos):
    outputs = model.run_method(
        "forward",
        (
            torch.tensor([ids], dtype=torch.int64),
            torch.tensor([pos], dtype=torch.int64),
        ),
    )
    result = outputs[0].clone()
    assert result.shape == (1, 128000) and result.dtype == torch.float32
    assert torch.isfinite(result).all()
    return result[0]


ids = reference["input_ids"]
started = time.perf_counter()
logits = forward(ids, 0)
report["prefill_seconds"] = time.perf_counter() - started
ref = reference["logits"][0]
report["first_token"] = int(logits.argmax())
report["reference_first_token"] = int(ref.argmax())
report["cosine_similarity"] = float(
    torch.nn.functional.cosine_similarity(logits, ref, dim=0)
)
report["reference_top1_rank"] = int((logits > logits[int(ref.argmax())]).sum()) + 1
report["max_abs_error"] = float((logits - ref).abs().max())
print("PARITY", json.dumps(report), flush=True)
assert report["reference_top1_rank"] <= 5, report
tokens = []
pos = len(ids)
start = time.perf_counter()
for step in range(args.steps):
    token = int(logits.argmax())
    if token == 124900:
        break
    tokens.append(token)
    print(
        "TOKEN",
        step,
        token,
        repr(tokenizer.decode([token], skip_special_tokens=False)),
        flush=True,
    )
    logits = forward([token], pos)
    pos += 1
report["decode_tokens_per_second"] = len(tokens) / max(
    time.perf_counter() - start, 1e-6
)
# Exercise the upper prefill bound as well as single-token decoding.
forward([597] * 128, pos)
report["prefill_bound_checked"] = 128
report["generated_text"] = tokenizer.decode(tokens, skip_special_tokens=False)
print("RESULT", json.dumps(report, ensure_ascii=False), flush=True)
(folder / "validation.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
)
