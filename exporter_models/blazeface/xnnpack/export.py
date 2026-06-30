import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_PKG = _MODEL_DIR / "_blazeface"


def _load_blazeface():
    spec = importlib.util.spec_from_file_location(
        "blazeface_pkg", _PKG / "__init__.py",
        submodule_search_locations=[str(_PKG)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["blazeface_pkg"] = mod
    spec.loader.exec_module(mod)
    return mod


class _BlazeFaceDecoded(nn.Module):
    """Back model (256x256, 896 anchors) with anchor-decoding math baked in.
    Returns (boxes[896,4] xyxy, scores[896], keypoints[896,6,3])."""

    def __init__(self, model, anchors):
        super().__init__()
        self.model = model
        self.register_buffer("anchors", anchors)
        self.x_scale = model.x_scale
        self.y_scale = model.y_scale
        self.w_scale = model.w_scale
        self.h_scale = model.h_scale

    def forward(self, x):
        raw_regs, raw_clss = self.model(x)
        dx, dy, dw, dh = raw_regs[..., 0], raw_regs[..., 1], raw_regs[..., 2], raw_regs[..., 3]
        ax, ay, aw, ah = self.anchors[:, 0], self.anchors[:, 1], self.anchors[:, 2], self.anchors[:, 3]

        x_center = dx * aw + ax * self.x_scale
        y_center = dy * ah + ay * self.y_scale
        w = dw * aw
        h = dh * ah
        boxes = torch.stack(
            [x_center - w / 2.0, y_center - h / 2.0, x_center + w / 2.0, y_center + h / 2.0],
            dim=-1,
        )

        kps = []
        for k in range(6):
            off = 4 + k * 2
            kx = raw_regs[..., off] * aw + ax * self.x_scale
            ky = raw_regs[..., off + 1] * ah + ay * self.y_scale
            kps.append(torch.stack([kx, ky, torch.ones_like(kx)], dim=-1))
        keypoints = torch.stack(kps, dim=-2)

        return (
            boxes.squeeze(0),
            torch.sigmoid(raw_clss).squeeze(0).squeeze(-1),
            keypoints.squeeze(0),
        )


def export(config: dict, out_dir: str) -> str:
    import numpy as np
    from torch.export import export as torch_export

    bf = _load_blazeface()
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    eager = bf.BlazeFace(back_model=True)
    eager.load_weights(str(_PKG / "blazefaceback.pth"))
    eager.eval()
    anchors = torch.tensor(np.load(str(_PKG / "anchorsback.npy")), dtype=torch.float32)
    model = _BlazeFaceDecoded(eager, anchors).eval()

    inputs = (torch.randn(*shape, dtype=torch.float32),)
    ep = torch_export(model, inputs, strict=True)

    from exporter.backends import lower
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
