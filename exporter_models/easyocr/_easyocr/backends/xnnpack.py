"""XNNPACK (CPU) backend helpers for the EasyOCR exporter.

True dynamic CPU delegate. CRAFT int8; CRNN fixed-width int8 (scan-LSTM blocks dynamic int8).

  InsertContig          ExportPass: contiguous() after every permute/transpose.
  partitioner()         XnnpackPartitioner with MaxDimConfig stripped + ViewCopyConfig.
  pt2e_int8(...)        static-int8 quantize primitive (one method, one shot).
"""
import torch
from torch.export import export as te
from executorch.exir.pass_base import ExportPass


class InsertContig(ExportPass):
    """Insert aten.contiguous after every permute/transpose/t (XNNPACK rejects 0-stride views)."""
    _OPS = {torch.ops.aten.permute.default, torch.ops.aten.permute_copy.default,
            torch.ops.aten.transpose.int, torch.ops.aten.transpose_copy.int,
            torch.ops.aten.t.default, torch.ops.aten.t_copy.default}

    def call_operator(self, op, args, kwargs, meta):
        r = super().call_operator(op, args, kwargs, meta)
        if op in self._OPS:
            r = super().call_operator(torch.ops.aten.contiguous.default, (r,), {}, meta)
        return r


def partitioner():
    """XnnpackPartitioner with MaxDimConfig removed + ViewCopyConfig added (CRAFT upsample/permutes)."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    try:
        from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS, MaxDimConfig
        from executorch.backends.xnnpack.partition.config.generic_node_configs import ViewCopyConfig
        configs = [c for c in ALL_PARTITIONER_CONFIGS if c is not MaxDimConfig] + [ViewCopyConfig]
        return XnnpackPartitioner(configs=configs)
    except Exception:
        return XnnpackPartitioner()


def pt2e_int8(module_or_factory, example_inputs, calib_inputs, *,
              is_per_channel=True, dynamic_shapes=None):
    """PT2E static int8 for ONE method: trace -> prepare -> calibrate -> convert.

    module_or_factory: nn.Module or 0-arg factory (use a factory per bucket — convert_pt2e
    folds weights in place). is_per_channel: True CRNN, False CRAFT. dynamic_shapes: {axis:Dim} or None.
    """
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    module = module_or_factory() if callable(module_or_factory) else module_or_factory
    ds = ({**dynamic_shapes},) if dynamic_shapes else None
    qcfg = get_symmetric_quantization_config(is_per_channel=is_per_channel)
    gm = te(module, example_inputs, dynamic_shapes=ds, strict=False).module()
    gm = prepare_pt2e(gm, XNNPACKQuantizer().set_global(qcfg))
    for c in calib_inputs:
        gm(*c) if isinstance(c, tuple) else gm(c)
    gm = convert_pt2e(gm)
    return te(gm, example_inputs, dynamic_shapes=ds, strict=False)
