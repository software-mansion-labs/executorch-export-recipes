"""Instance segmentation model wrappers.

Contract: (bbox [B,Q,4] x1y1x2y2 pixel, scores [B,Q,2] (conf,class_id), mask_logits [B,Q,H,W] pre-sigmoid)
"""

from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class InstanceSegmentationWrapper(nn.Module):
    """Abstract base — returns (bbox [Q,4], scores [Q], classes [Q], masks [Q,H,W])."""

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...


class YOLOInstanceSegmentation(InstanceSegmentationWrapper):
    """Wraps YOLO segmentation model.

    Uses default postprocess (topk max_det=300), computes mask_logits
    via bmm with protos at native resolution (or optionally downscaled).
    """

    def __init__(
        self,
        yolo_inner_model: nn.Module,
        method_size: int,
        mask_size: int | None = None,
    ) -> None:
        super().__init__()
        self.inner = yolo_inner_model

        self.inner.eval()
        self.inner.model[-1].export = True

        with torch.no_grad():
            preds, protos = self.inner(torch.randn(1, 3, method_size, method_size))

        self.nm: int = protos.shape[1]
        self.mask_size = mask_size
        proto_h, proto_w = protos.shape[2], protos.shape[3]
        effective = (
            f"{mask_size}x{mask_size}" if mask_size else f"{proto_h}x{proto_w} (native)"
        )
        print(
            f"  preds {preds.shape}, protos {protos.shape}, nm={self.nm}, mask_size={effective}"
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        preds, protos = self.inner(x)

        bbox = preds[0, :, :4]
        scores = preds[0, :, 4]
        classes = preds[0, :, 5]
        # The YOLO26 end2end head returns preds in fp32 even on a halved model,
        # while protos follow the model dtype. Align them so the bmm below works
        # under an fp16 export; this is a no-op when both are already fp32.
        mask_coeffs = preds[:, :, -self.nm :].to(protos.dtype)

        if self.mask_size is not None:
            protos = F.interpolate(
                protos,
                size=(self.mask_size, self.mask_size),
                mode="bilinear",
                align_corners=False,
            )

        B, C, H, W = protos.shape
        mask_logits = torch.bmm(mask_coeffs, protos.reshape(B, C, H * W)).reshape(
            B, -1, H, W
        )
        masks = mask_logits[0]

        return bbox, scores, classes, masks
