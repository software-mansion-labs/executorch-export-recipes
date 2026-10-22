"""Backend-agnostic lowering / serialization for the PaddleOCR exporters.

Build the edge program, print the per-method delegation report, write the .pte.
Backend choice enters only as the `partitioners` list (built in `backends.*`).
"""
import os
import time

from executorch.exir import (EdgeCompileConfig, ExecutorchBackendConfig,
                             to_edge_transform_and_lower)
from executorch.devtools.backend_debug import get_delegation_info

# Register quantized_decomposed.*.out so int8 ops left in the portable graph can lower
# (needed for int8 + dynamic). Skip via ET_SKIP_QUANTIZED_KERNELS=1 if core already
# registers them (re-registering hard-aborts uncatchably). Gate BEFORE the import.
import os as _os
if _os.environ.get("ET_SKIP_QUANTIZED_KERNELS") != "1":
    try:
        import executorch.kernels.quantized  # noqa: F401
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PTE_DIR = os.path.join(HERE, "pte")   # all exported .pte land here (keep ppocr/ clean)


def pte_path(name):
    """Resolve a bare name (may include a backend subdir) to <pte>/<name>.pte."""
    p = os.path.join(PTE_DIR, name if name.endswith(".pte") else f"{name}.pte")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def enum_constant_methods(enum_map):
    """Bake per-method `<method>_enum_shapes` ([N,4] NCHW) for CoreML enumerated; the
    RNE C++ reads them to snap each stage's input. Per-method avoids fused-PTE collision.
    enum_map: {mname: {in_name: [[1,c,h,w],...]}}."""
    if not enum_map:
        return None
    import torch
    out = {f"{mname}_enum_shapes": torch.tensor(next(iter(in_map.values())),
                                                dtype=torch.int32)
           for mname, in_map in enum_map.items()}
    # Legacy global `enum_shapes` (first method's) for un-qualified single-enum lookups.
    out["enum_shapes"] = next(iter(out.values()))
    return out


def _resolve(out):
    """Absolute path verbatim; bare name -> <ppocr>/pte/<name>.pte."""
    if os.path.isabs(out):
        return out
    return pte_path(out)


def report_delegation(low, names, *, width=16):
    """Print delegated/total node counts (+ lowered blobs) per method."""
    for name in names:
        gm = low.exported_program(name).graph_module
        info = get_delegation_info(gm)
        tot = info.num_delegated_nodes + info.num_non_delegated_nodes
        blobs = sum(1 for n in gm.graph.nodes
                    if n.op == "get_attr" and "lowered_module" in str(n.target))
        print(f"    {name:{width}s}: {info.num_delegated_nodes}/{tot} "
              f"({info.num_delegated_nodes/max(tot,1)*100:.1f}%)  blobs={blobs}")


def save_pte(low, out, *, memory_planning=None):
    """to_executorch + write; returns the written path."""
    cfg = dict(extract_delegate_segments=True)
    if memory_planning is not None:
        cfg["memory_planning_pass"] = memory_planning
    prog = low.to_executorch(ExecutorchBackendConfig(**cfg))
    path = _resolve(out)
    with open(path, "wb") as f:
        prog.write_to_file(f)
    print(f"    SAVED {path}  ({os.path.getsize(path)/1e6:.2f} MB)")
    return path


def lower_save(programs, partitioners, out, *, transform_passes=None,
               compile_config=None, enum_map=None, extra_constants=None,
               memory_planning=None, report=True, width=16):
    """Lower `programs` ({method: ep}) with `partitioners`, report, write the .pte.
    `extra_constants` merge into the baked constant_methods (e.g. is_bucketed);
    `memory_planning` is passed through to save_pte (e.g. RT-DETR layout)."""
    t0 = time.time()
    cc = compile_config or EdgeCompileConfig(_check_ir_validity=False)
    consts = enum_constant_methods(enum_map) or {}
    if extra_constants:
        consts.update(extra_constants)
    low = to_edge_transform_and_lower(
        programs, partitioner=partitioners,
        constant_methods=(consts or None),
        compile_config=cc, transform_passes=transform_passes)
    print(f"    lowered {out} ({time.time()-t0:.1f}s)")
    if report:
        report_delegation(low, list(programs), width=width)
    return save_pte(low, out, memory_planning=memory_planning)
