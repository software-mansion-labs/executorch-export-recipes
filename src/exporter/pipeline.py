import time
from dataclasses import dataclass
from pathlib import Path

from .backends import default_atol
from .errors import IOContractError
from .pte_inspect import read_pte_io
from .runner import run_export
from .spec import Method, Spec, Variant


@dataclass
class IOCheck:
    method: str
    ok: bool
    deltas: list[str]


@dataclass
class VariantResult:
    file: str
    precision: str
    io_ok: bool
    io_deltas: list[str]
    parity_ok: bool | None
    bench_ms: float | None


@dataclass
class ExportPlan:
    model: str
    backend: str
    export_script: str
    variants: list[str]
    output_dir: str
    atol: float


def is_runtime_verifiable(spec: Spec, variant: Variant) -> bool:
    if spec.backend == "mlx":
        return False
    if variant.file is None:  # multi-component
        return False
    if "text-generation" in spec.capabilities:  # LLM: forward can't be zero-probed
        return False
    return "forward" in variant.methods


def _select(spec: Spec, which: str | None) -> list[Variant]:
    if which is None:
        return [spec.default_variant()]
    if which == "all":
        return list(spec.variants)
    return [spec.variant_by_precision(which)]


def plan(spec: Spec, *, export_script: str, which: str | None, output_dir: str, atol: float | None) -> ExportPlan:
    variants = _select(spec, which)
    quantized = any(v.quantized for v in variants)
    resolved_atol = atol if atol is not None else default_atol(spec.backend, quantized)
    return ExportPlan(
        model=spec.model,
        backend=spec.backend,
        export_script=str(export_script),
        variants=[v.file for v in variants],
        output_dir=output_dir,
        atol=resolved_atol,
    )


def compare_io(expected: dict[str, Method], actual: dict[str, dict]) -> IOCheck:
    deltas: list[str] = []
    for mname, method in expected.items():
        if mname not in actual:
            deltas.append(f"method '{mname}' missing from exported .pte")
            continue
        for kind, exp_list in (("input", method.inputs), ("output", method.outputs)):
            act_list = actual[mname][f"{kind}s"]
            if len(exp_list) != len(act_list):
                deltas.append(f"{mname} {kind} count {len(exp_list)}!={len(act_list)}")
                continue
            for i, (e, a) in enumerate(zip(exp_list, act_list)):
                if list(e.shape) != list(a.shape):
                    deltas.append(f"{mname} {kind}[{i}] shape {e.shape}!={a.shape}")
                if e.dtype != a.dtype:
                    deltas.append(f"{mname} {kind}[{i}] dtype {e.dtype}!={a.dtype}")
    return IOCheck(method="all", ok=not deltas, deltas=deltas)


def export_variant(spec: Spec, variant: Variant, *, export_script: str, output_dir: str, atol: float, run_bench: bool = True) -> VariantResult:
    config_dict = {
        "model": spec.model,
        "backend": spec.backend,
        "variant": {
            "file": variant.file,
            "precision": variant.precision,
            "quantized": variant.quantized,
            "methods": {
                name: {
                    "inputs": [{"shape": t.shape, "dtype": t.dtype} for t in m.inputs],
                    "outputs": [{"shape": t.shape, "dtype": t.dtype} for t in m.outputs],
                }
                for name, m in variant.methods.items()
            },
        },
    }

    pte_path = run_export(export_script, config_dict, output_dir)

    actual_io = read_pte_io(pte_path)  # all methods, no filter
    io_check = compare_io(variant.methods, actual_io)
    if not io_check.ok:
        raise IOContractError(
            f"{spec.model}/{variant.file}: authored I/O != exported: " + "; ".join(io_check.deltas)
        )

    parity_ok, bench_ms = None, None
    if is_runtime_verifiable(spec, variant):
        parity_ok, bench_ms = _runtime_sanity(pte_path, variant.methods["forward"], run_bench)

    return VariantResult(
        file=variant.file, precision=variant.precision,
        io_ok=True, io_deltas=[], parity_ok=parity_ok, bench_ms=bench_ms,
    )


def _torch_dtype(name: str):
    import torch
    return {
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bool": torch.bool,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _runtime_sanity(pte_path: str, method: Method, run_bench: bool):
    import torch
    from executorch.runtime import Runtime
    inputs = [torch.zeros(tuple(t.shape), dtype=_torch_dtype(t.dtype)) for t in method.inputs]
    runtime = Runtime.get()
    loaded = runtime.load_program(str(pte_path))
    fwd = loaded.load_method("forward")
    out = fwd.execute(inputs)[0]
    parity_ok = bool(torch.isfinite(out).all())
    bench_ms = None
    if run_bench:
        tik = time.time()
        for _ in range(100):
            fwd.execute(inputs)
        bench_ms = (time.time() - tik) / 100 * 1000
    return parity_ok, bench_ms
