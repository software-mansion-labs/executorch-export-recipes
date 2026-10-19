from .errors import BackendError


def lower(
    exported_program,
    backend: str,
    constant_methods: dict | None = None,
    transform_passes: list | None = None,
):
    """`transform_passes` run on the edge program before partitioning (e.g. a
    constant-folding pass), see `to_edge_transform_and_lower`."""
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower

        return to_edge_transform_and_lower(
            exported_program,
            transform_passes=transform_passes,
            partitioner=[XnnpackPartitioner()],
            constant_methods=constant_methods,
        ).to_executorch()
    raise BackendError(f"lower() is the shared xnnpack helper; '{backend}' models lower inline in their export.py")


# The MLX delegate serializes its graph as a FlatBuffer union whose member order
# IS the wire format. software-mansion-labs/executorch (the fork the shipped
# MLXBackend runtime is built from, branch @ms/separate-backends-1.4.1) carries
# `CumMaxNode` at union value 112, which upstream does not have, so every op from
# there on is shifted by one against the PyPI `executorch` wheel. A .pte written
# by the wheel therefore tags its quantized matmuls 121 where the runtime reads
# 121 as ArgPartition, and loading it fails with `Error::InvalidProgram` -- no
# version check catches it, because both sides write schema version "1".
_MLX_FORK_ANCHOR = (112, "CumMaxNode")


def require_mlx_runtime_fork() -> None:
    """Fail the export unless the env's MLX backend is the runtime-locked fork."""
    from executorch.backends.mlx.serialization._generated_serializers import (
        MLX_OP_TYPE_NAMES,
    )

    value, name = _MLX_FORK_ANCHOR
    if MLX_OP_TYPE_NAMES.get(value) != name:
        raise BackendError(
            f"MLX op union value {value} is {MLX_OP_TYPE_NAMES.get(value)!r}, expected "
            f"{name!r}: this env's executorch is not the runtime-locked fork "
            "(software-mansion-labs/executorch @ @ms/separate-backends-1.4.1). "
            "A .pte exported here would fail to load with Error::InvalidProgram. "
            "Overlay the fork's backends/mlx into the env and regenerate the "
            "bindings -- see the model's NOTES.md."
        )
