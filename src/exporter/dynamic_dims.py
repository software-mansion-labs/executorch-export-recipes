"""Build the ``get_dynamic_dims_<method>`` companion program.

The react-native-executorch runtime reads a model's dynamic input-shape bounds
from a companion method named ``get_dynamic_dims_<methodName>`` (e.g.
``get_dynamic_dims_forward``). It takes no arguments and returns one int32
``[rank, 3]`` tensor per tensor input of the target method, each row holding the
``[min, max, step]`` bounds for the corresponding dimension. Static dimensions
are encoded as ``[n, n, 1]``. See ``ModelHostObject`` in the native runtime for
the full contract.
"""

import torch
from torch.export import export as torch_export


def build_dynamic_dims_program(input_bounds):
    """Export the companion program advertising per-input dynamic bounds.

    Args:
        input_bounds: one entry per tensor input of the target method, each a
            list of ``[min, max, step]`` rows (one row per dimension of that
            input, in order).

    Returns:
        An ``ExportedProgram`` whose no-arg ``forward`` returns one int32
        ``[rank, 3]`` tensor per input, ready to register under
        ``get_dynamic_dims_<methodName>``.
    """
    bounds = [torch.tensor(rows, dtype=torch.int32) for rows in input_bounds]

    class _DynamicDims(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            self._count = len(tensors)
            for i, t in enumerate(tensors):
                self.register_buffer(f"bounds_{i}", t)

        def forward(self):
            return tuple(getattr(self, f"bounds_{i}").clone() for i in range(self._count))

    return torch_export(_DynamicDims(bounds), ())
