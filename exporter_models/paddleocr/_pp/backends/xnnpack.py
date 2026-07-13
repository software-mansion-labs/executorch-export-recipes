"""XNNPACK (CPU) backend helpers. True-dynamic; precision lever is int8 (PT2E static,
real-crop calibrated). fp16 is not a real lever here (compute-only)."""
import torch
from torch.export import export as te
from executorch.exir.pass_base import ExportPass


class InsertContig(ExportPass):
    """Insert aten.contiguous after permute/transpose/t (XNNPACK rejects 0-stride views)."""
    _OPS = {torch.ops.aten.permute.default, torch.ops.aten.permute_copy.default,
            torch.ops.aten.transpose.int, torch.ops.aten.transpose_copy.int,
            torch.ops.aten.t.default, torch.ops.aten.t_copy.default}

    def call_operator(self, op, args, kwargs, meta):
        r = super().call_operator(op, args, kwargs, meta)
        if op in self._OPS:
            r = super().call_operator(torch.ops.aten.contiguous.default, (r,), {}, meta)
        return r


def partitioner():
    """Plain XNNPACK partitioner (det/rec, orientation, table, dewarp, fused)."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    return XnnpackPartitioner()


def layout_partitioner(quantize):
    """RT-DETR (PP-DocLayoutV3) partitioner: add ViewCopyConfig (if absent) so the
    ~530 deformable-attn view_copy delegate; DynamicQuantLinearConfig only for torchao paths."""
    from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS
    from executorch.backends.xnnpack.partition.config.generic_node_configs import ViewCopyConfig
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    cfgs = list(ALL_PARTITIONER_CONFIGS)
    if ViewCopyConfig not in cfgs:
        cfgs.append(ViewCopyConfig)
    kw = {"configs": cfgs}
    if quantize in ("int8wo", "8da4w"):
        try:
            from executorch.backends.xnnpack.partition.config.gemm_configs import DynamicQuantLinearConfig
            kw["per_op_mode_config"] = {DynamicQuantLinearConfig}
        except ImportError:
            pass
    return XnnpackPartitioner(**kw)


def _prepare_qd8_all_convs(gm, qcfg):
    """prepare_pt2e with conv->relu convs INCLUDED under dynamic (qd8) quant.

    The stock annotator defers relu-followed convs to the `conv_relu` pattern, but
    that pattern is disabled for dynamic configs (SUPPORTED_PATTERNS is_dynamic=False)
    — so every conv->relu would silently stay fp32. A qd8 conv's output is fp32
    anyway, so annotating the conv alone is exact; all relu-consuming patterns are
    inert under a dynamic config, making it safe to neutralize is_relu_node for the
    duration of prepare (annotate) only.
    """
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e
    from executorch.backends.xnnpack.quantizer import xnnpack_quantizer_utils as QU
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer
    orig = QU.is_relu_node
    QU.is_relu_node = lambda n: False
    try:
        return prepare_pt2e(gm, XNNPACKQuantizer().set_global(qcfg))
    finally:
        QU.is_relu_node = orig


def pt2e_int8(module_or_factory, example_inputs, calib_inputs, *,
              is_per_channel=True, is_dynamic=False, dynamic_shapes=None):
    """PT2E int8 for one method: trace -> prepare -> calibrate -> convert.

    `is_dynamic=False` = STATIC int8 (int8 activations from calibration) — max shrink
    but lossy on regressors/AR. `is_dynamic=True` = qd8 (int8 weights, activations
    quantized per-input at runtime) — gentler, lossless here, but only ~halves size
    (conv weights it can't qd8 stay fp32).

    `module_or_factory`: nn.Module, or a 0-arg factory (pass a factory per bucket —
    convert_pt2e folds weights in place, reuse corrupts). `calib_inputs` must be real
    activations. `dynamic_shapes`: {axis: Dim} or None.
    """
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    # Module instances are callable too, so dispatch on type not callable().
    module = (module_or_factory if isinstance(module_or_factory, torch.nn.Module)
              else module_or_factory())
    ds = ({**dynamic_shapes},) if dynamic_shapes else None
    qcfg = get_symmetric_quantization_config(is_per_channel=is_per_channel,
                                             is_dynamic=is_dynamic)
    gm = te(module, example_inputs, dynamic_shapes=ds, strict=False).module()
    if is_dynamic:
        gm = _prepare_qd8_all_convs(gm, qcfg)
    else:
        gm = prepare_pt2e(gm, XNNPACKQuantizer().set_global(qcfg))
    for c in calib_inputs:
        gm(*c) if isinstance(c, tuple) else gm(c)
    gm = convert_pt2e(gm)
    return te(gm, example_inputs, dynamic_shapes=ds, strict=False)
