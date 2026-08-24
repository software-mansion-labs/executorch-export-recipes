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


def _prepare_qd8_all_convs(gm, qcfg, exclude_modules=()):
    """prepare_pt2e with conv->relu convs INCLUDED under dynamic (qd8) quant.

    The stock annotator defers relu-followed convs to the `conv_relu` pattern, but
    that pattern is disabled for dynamic configs (SUPPORTED_PATTERNS is_dynamic=False)
    — so every conv->relu would silently stay fp32 (CRAFT/VGG: ~2/3 of the weights).
    A qd8 conv's output is fp32 anyway, so annotating the conv alone is exact; all
    relu-consuming patterns are inert under a dynamic config, making it safe to
    neutralize is_relu_node for the duration of prepare (annotate) only.

    exclude_modules: nn_module_stack path substrings whose annotations are stripped
    after the base annotate — those modules stay fp32 (accuracy-critical regions;
    qd8 noise compounds with depth, see the builder's per-slice findings).
    """
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e
    from torchao.quantization.pt2e.quantizer import Quantizer
    from executorch.backends.xnnpack.quantizer import xnnpack_quantizer_utils as QU
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer

    def stack_path(n):
        s = n.meta.get("nn_module_stack")
        return list(s.values())[-1][0] if s else ""

    class _Filtered(Quantizer):
        def __init__(self, base):
            super().__init__(); self.base = base
        def annotate(self, model):
            model = self.base.annotate(model)
            for n in model.graph.nodes:
                if ("quantization_annotation" in n.meta
                        and any(s in stack_path(n) for s in exclude_modules)):
                    del n.meta["quantization_annotation"]
            return model
        def validate(self, model):
            return self.base.validate(model)

    orig = QU.is_relu_node
    QU.is_relu_node = lambda n: False
    try:
        return prepare_pt2e(gm, _Filtered(XNNPACKQuantizer().set_global(qcfg)))
    finally:
        QU.is_relu_node = orig


def pt2e_int8(module_or_factory, example_inputs, calib_inputs, *,
              is_per_channel=True, is_dynamic=False, dynamic_shapes=None,
              exclude_modules=()):
    """PT2E int8 for ONE method: trace -> prepare -> calibrate -> convert.

    `is_dynamic=False` = STATIC int8 (activation scales from calibration) — max shrink
    but the scales don't transfer across dynamic shapes/content. `is_dynamic=True` =
    qd8 (int8 weights, activations quantized per-input at runtime) — the only int8
    that stays sane under dynamic_shapes; covers non-depthwise conv2d + linear.

    module_or_factory: nn.Module or 0-arg factory (use a factory per bucket — convert_pt2e
    folds weights in place). is_per_channel: True CRNN, False CRAFT. dynamic_shapes: {axis:Dim} or None.
    """
    import torch
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
        gm = _prepare_qd8_all_convs(gm, qcfg, exclude_modules)
    else:
        gm = prepare_pt2e(gm, XNNPACKQuantizer().set_global(qcfg))
    for c in calib_inputs:
        gm(*c) if isinstance(c, tuple) else gm(c)
    gm = convert_pt2e(gm)
    return te(gm, example_inputs, dynamic_shapes=ds, strict=False)
