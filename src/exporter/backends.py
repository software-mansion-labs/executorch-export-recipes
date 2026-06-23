from .errors import BackendError

SUPPORTED_BACKENDS = {"xnnpack", "coreml", "mlx"}


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
    if backend in {"coreml", "mlx"}:
        raise BackendError(f"backend '{backend}' not yet implemented in this harness")
    raise BackendError(f"unknown backend '{backend}' (supported: {', '.join(sorted(SUPPORTED_BACKENDS))})")
