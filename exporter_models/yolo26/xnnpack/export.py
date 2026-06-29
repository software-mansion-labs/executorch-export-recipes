from pathlib import Path

import torch
from torch.export import export as torch_export

from exporter.backends import lower


class _YoloDetection(torch.nn.Module):
    """Repo YOLOObjectDetection contract: packed [max_det,6] head -> (boxes, scores, labels)."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.inner.eval()
        self.inner.model[-1].export = True

    def forward(self, x):
        preds = self.inner(x)
        p = preds[0]
        return p[:, :4], p[:, 4], p[:, 5]


def export(config: dict, out_dir: str) -> str:
    from ultralytics import YOLO

    size = config.get("size") or "n"
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    yolo = YOLO(f"yolo26{size}.pt")
    model = _YoloDetection(yolo.model).eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
