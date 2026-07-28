from .errors import BackendError


def lower(exported_program, backend: str, constant_methods: dict | None = None):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        return to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
            constant_methods=constant_methods,
        ).to_executorch()
    raise BackendError(
        f"lower() is the shared xnnpack helper; '{backend}' models lower inline in their export.py"
    )
