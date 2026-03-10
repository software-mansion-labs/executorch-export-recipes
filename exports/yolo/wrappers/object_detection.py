"""Object detection model wrappers.

Contract: (bbox [Q,4] x1y1x2y2 pixel, scores [Q], labels [Q])
"""

from abc import abstractmethod

import torch
from torch import Tensor, nn


class ObjectDetectionWrapper(nn.Module):
    """Abstract base — returns (bbox [Q,4], scores [Q], labels [Q])."""

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...


class YOLOObjectDetection(ObjectDetectionWrapper):
    """Wraps YOLO detection model.

    head.export=True gives a packed [1, max_det, 6] tensor.
    Squeezes batch dim to match RF-DETR contract.
    """

    def __init__(self, yolo_inner_model: nn.Module, method_size: int = 640) -> None:
        super().__init__()
        self.inner = yolo_inner_model
        self.inner.eval()
        self.inner.model[-1].export = True

        with torch.no_grad():
            preds = self.inner(torch.randn(1, 3, method_size, method_size))

        if not isinstance(preds, Tensor) or preds.ndim != 3 or preds.shape[-1] != 6:
            raise ValueError(f"Expected [B,Q,6], got {preds.shape}")

        print(f"  preds {preds.shape}")

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        preds = self.inner(x)
        # preds[0]: [max_det, 6] — [:4] x1y1x2y2 pixel, [4] conf, [5] class_id
        p = preds[0]
        return p[:, :4], p[:, 4], p[:, 5]
