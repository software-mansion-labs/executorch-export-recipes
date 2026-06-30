"""Thin nn.Module wrappers shared by the PaddleOCR exporters.

The HF models return a `*ModelOutput` whose `.last_hidden_state` is the tensor we
actually export. These wrappers unwrap it (and, for the fp16 graph, cast at the I/O
boundary so the program takes/returns fp32 while computing in half).
"""
import torch
import torch.nn as nn


class Norm(nn.Module):
    """Bake input normalization INTO the graph so the runtime feeds raw [0,1] RGB and does no
    per-family norm — unifies the EasyOCR/PaddleOCR detector input contract. Wraps the HF model;
    returns its ModelOutput so Stage(.last_hidden_state) still works."""
    def __init__(self, m, mean, std):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return self.m((x - self.mean) / self.std)


class Stage(nn.Module):
    """fp32 / int8 pass-through: HF model -> bare last_hidden_state tensor."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).last_hidden_state


class StageFp16(nn.Module):
    """fp16 compute with an fp32 I/O boundary (XNNPACK/Vulkan hand .half() graph)."""
    def __init__(self, m):
        super().__init__()
        self.m = m.half()

    def forward(self, x):
        return self.m(x.half()).last_hidden_state.float()


class HalfWrap(nn.Module):
    """fp16 graph for an arbitrary multi-input module: half the module, cast every
    input, float every output (tuple-aware). Used by the fused-helper export."""
    def __init__(self, mod):
        super().__init__()
        self.mod = mod.half()

    def forward(self, *xs):
        out = self.mod(*[x.half() for x in xs])
        if isinstance(out, tuple):
            return tuple(o.float() for o in out)
        return out.float()
