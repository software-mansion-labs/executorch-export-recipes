import time
from dataclasses import dataclass
from pathlib import Path

from .backends import default_atol, lower
from .errors import IOContractError, ParityError
from .pte_inspect import read_pte_io
from .resolver import resolve_wrapper
from .spec import Method, Spec, Variant
from .wrappers import get_builder


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
    parity_max_diff: float | None
    parity_ok: bool
    bench_ms: float | None


@dataclass
class ExportPlan:
    model: str
    source: str
    backend: str
    wrapper: str
    wrapper_reason: str
    wrapper_kwargs: dict
    variants: list[str]
    output_dir: str
    atol: float


def _select(spec: Spec, which: str | None) -> list[Variant]:
    if which is None:
        return [spec.default_variant()]
    if which == "all":
        return list(spec.variants)
    return [spec.variant_by_precision(which)]


def plan(spec: Spec, *, which: str | None, output_dir: str, atol: float | None) -> ExportPlan:
    res = resolve_wrapper(spec.source)
    variants = _select(spec, which)
    quantized = any(v.quantized for v in variants)
    resolved_atol = atol if atol is not None else default_atol(spec.backend, quantized)
    return ExportPlan(
        model=spec.model,
        source=spec.source,
        backend=spec.backend,
        wrapper=res.wrapper,
        wrapper_reason=res.reason,
        wrapper_kwargs=res.kwargs,
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


def example_inputs_and_shapes(method: Method):
    import torch
    from torch.export import Dim
    inputs = tuple(torch.zeros(tuple(t.shape), dtype=torch.int64) for t in method.inputs)
    max_tokens = method.inputs[0].shape[-1]
    dim = Dim("tokens", min=1, max=max_tokens)
    dynamic_shapes = {f"arg{i}": {len(t.shape) - 1: dim} for i, t in enumerate(method.inputs)}
    # torch.export keys dynamic_shapes by positional arg name; build a positional tuple instead.
    dynamic = tuple({len(t.shape) - 1: dim} for t in method.inputs)
    return inputs, dynamic


def export_variant(spec: Spec, variant: Variant, *, output_dir: str, atol: float, run_bench: bool = True) -> VariantResult:
    import torch
    from torch.export import export

    res = resolve_wrapper(spec.source)
    module = get_builder(res.wrapper)(spec.source, variant, **res.wrapper_kwargs)
    module.eval()

    method = variant.methods["forward"]
    inputs, dynamic = example_inputs_and_shapes(method)

    with torch.no_grad():
        eager_out = module(*inputs)

    ep = export(module, inputs, dynamic_shapes=dynamic)
    program = lower(ep, spec.backend)

    out_path = Path(output_dir) / variant.file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)

    actual_io = read_pte_io(out_path, "forward")
    io_check = compare_io({"forward": method}, actual_io)
    if not io_check.ok:
        raise IOContractError(
            f"{spec.model}/{variant.file}: authored I/O != exported: " + "; ".join(io_check.deltas)
        )

    from executorch.runtime import Runtime
    runtime = Runtime.get()
    loaded = runtime.load_program(str(out_path))
    fwd = loaded.load_method("forward")
    rt_out = fwd.execute(list(inputs))[0]
    max_diff = float((eager_out - rt_out).abs().max())
    parity_ok = max_diff <= atol
    if not parity_ok:
        raise ParityError(
            f"{spec.model}/{variant.file}: allclose failed: max abs diff {max_diff:.2e} > atol {atol:.0e} (pass --atol to relax)"
        )

    bench_ms = None
    if run_bench:
        tik = time.time()
        for _ in range(100):
            fwd.execute(list(inputs))
        bench_ms = (time.time() - tik) / 100 * 1000

    return VariantResult(
        file=variant.file, precision=variant.precision,
        io_ok=True, io_deltas=[], parity_max_diff=max_diff, parity_ok=True, bench_ms=bench_ms,
    )
