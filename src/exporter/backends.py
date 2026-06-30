from .errors import BackendError
from .spec import _BACKENDS as SUPPORTED_BACKENDS


def default_atol(backend: str, quantized: bool) -> float:
    if backend not in SUPPORTED_BACKENDS:
        raise BackendError(f"unknown backend '{backend}' (supported: {', '.join(sorted(SUPPORTED_BACKENDS))})")
    return 1e-2 if quantized else 1e-4


def lower(exported_program, backend: str):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        return to_edge_transform_and_lower(
            exported_program, partitioner=[XnnpackPartitioner()]
        ).to_executorch()
    raise BackendError(
        f"lower() is the shared xnnpack helper; '{backend}' models lower inline in their export.py"
    )
