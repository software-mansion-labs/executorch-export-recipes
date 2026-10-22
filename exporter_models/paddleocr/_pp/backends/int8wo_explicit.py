"""Int8 weight-only quantization that emits explicit q/dq ops in the FX graph.

torchao's Int8WeightOnlyConfig uses tensor subclasses (AffineQuantizedTensor) that
decompose into `to(fp32) -> mul(scale) -> mm` during torch.export — XNNPACK's
partitioner can't recognize that as a quantized linear. This module replaces each
nn.Linear with one that calls `quantized_decomposed::dequantize_per_channel`
explicitly, which the partitioner matches as INT8_WEIGHT_ONLY (after our
delegate patch). XNNPACK then dispatches to its `f32_qc8w_gemm` microkernel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Side-effect: registers torch.ops.quantized_decomposed.dequantize_per_channel
import torch.ao.quantization.fx._decomposed  # noqa: F401


class Int8WOLinear(nn.Module):
    """nn.Linear with int8 per-channel weight and fp32 activation.

    Weight quant is symmetric, per-output-channel, zp=0. The forward calls the
    explicit dequant op so torch.export captures it as a single node.
    """

    def __init__(self, src: nn.Linear):
        super().__init__()
        w = src.weight.detach().to(torch.float32)
        # Per-output-channel absmax
        absmax = w.abs().amax(dim=1).clamp(min=1e-8)
        scales = absmax / 127.0
        q = (w / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        # zp must be int8 (matching weight dtype) so the XNNPACK partitioner's
        # _check_node_has_valid_dtype accepts the dequant inputs.
        zp = torch.zeros_like(scales, dtype=torch.int8)
        # Register as parameters so torch.export treats them as static weights
        self.weight_int8 = nn.Parameter(q, requires_grad=False)
        self.scales = nn.Parameter(scales.to(torch.float32), requires_grad=False)
        self.zero_points = nn.Parameter(zp, requires_grad=False)
        if src.bias is not None:
            self.bias = nn.Parameter(src.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None
        self.in_features = src.in_features
        self.out_features = src.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            self.weight_int8,
            self.scales,
            self.zero_points,
            0,         # axis
            -128,      # qmin
            127,       # qmax
            torch.int8,
        )
        # Bias is applied OUTSIDE the F.linear: XNNPACK's f32_qc8w
        # fully-connected path appears to ignore the bias tensor in the
        # current ExecuTorch bridge. Adding it as a separate op routes through
        # the standard add path which delegates correctly.
        out = F.linear(x, w, None)
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_linears_int8wo(
    module: nn.Module,
    *,
    skip_substrings: tuple[str, ...] = (),
) -> nn.Module:
    """In-place: replace every nn.Linear in `module` with Int8WOLinear.

    `skip_substrings`: any Linear whose qualified name contains one of these is
    left as fp32. Useful for layers known to be sensitive (e.g. lm_head, patch_embed).
    """
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if any(s in name for s in skip_substrings):
            continue
        parent_path, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        setattr(parent, attr, Int8WOLinear(child))
    return module
