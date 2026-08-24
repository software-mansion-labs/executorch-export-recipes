"""Backend-agnostic lowering / serialization for the EasyOCR exporter.

Build the edge program, report delegation, write the .pte. Backend enters only as
the `partitioners` argument (built in `backends.*`).
"""
import os
import time

from executorch.exir import (EdgeCompileConfig, ExecutorchBackendConfig,
                             to_edge_transform_and_lower)
from executorch.devtools.backend_debug import get_delegation_info

import os
# Register quantized_decomposed.*.out (needed for XNNPACK int8+dynamic). Skip via
# ET_SKIP_QUANTIZED_KERNELS=1 — a source-built core may already register them and
# re-registering hard-aborts (uncatchable C++ assert).
if os.environ.get("ET_SKIP_QUANTIZED_KERNELS") != "1":
    try:
        import executorch.kernels.quantized  # noqa: F401
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PTE_DIR = os.path.join(HERE, "pte")   # all exported .pte land here


def pte_path(name):
    """Resolve a bare name to <pte>/<name>.pte (name may include a backend subdir)."""
    p = os.path.join(PTE_DIR, name if name.endswith(".pte") else f"{name}.pte")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def enum_constant_methods(enum_map):
    """Bake per-method `<method>_enum_shapes` constant methods (C++ snaps input to these;
    absent => true-dynamic). enum_map: {mname: {in_name: [[1,c,h,w],...]}}."""
    if not enum_map:
        return None
    import torch
    out = {f"{mname}_enum_shapes": torch.tensor(next(iter(in_map.values())),
                                                dtype=torch.int32)
           for mname, in_map in enum_map.items()}
    # Legacy global `enum_shapes` (first method's) for un-qualified getEnumeratedShapes().
    out["enum_shapes"] = next(iter(out.values()))
    return out


def _resolve(out):
    if os.path.isabs(out):
        return out
    return pte_path(out)


def report_delegation(low, names, *, width=16):
    for name in names:
        gm = low.exported_program(name).graph_module
        info = get_delegation_info(gm)
        tot = info.num_delegated_nodes + info.num_non_delegated_nodes
        blobs = sum(1 for n in gm.graph.nodes
                    if n.op == "get_attr" and "lowered_module" in str(n.target))
        print(f"    {name:{width}s}: {info.num_delegated_nodes}/{tot} "
              f"({info.num_delegated_nodes/max(tot,1)*100:.1f}%)  blobs={blobs}")


def save_pte(low, out):
    prog = low.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=True))
    path = _resolve(out)
    with open(path, "wb") as f:
        prog.write_to_file(f)
    print(f"  SAVED: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def lower_save(programs, partitioners, out, *, transform_passes=None, enum_map=None,
               extra_constants=None, width=16):
    """Lower `programs` ({method: ep}) with `partitioners` (list OR {method: [part]}),
    report delegation, write the .pte. Returns the written path.
    `extra_constants` are merged into the baked constant_methods (e.g. is_bucketed)."""
    t0 = time.time()
    consts = enum_constant_methods(enum_map) or {}
    if extra_constants:
        consts.update(extra_constants)
    low = to_edge_transform_and_lower(
        programs, partitioner=partitioners,
        constant_methods=(consts or None),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        transform_passes=transform_passes)
    print(f"  lowered {out} ({time.time()-t0:.1f}s). DELEGATION:")
    report_delegation(low, list(programs), width=width)
    return save_pte(low, out)
