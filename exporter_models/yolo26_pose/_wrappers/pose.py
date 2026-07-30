"""YOLO Pose model wrappers.

Contract: (bbox [Q,4] x1y1x2y2 pixel, scores [Q], keypoints [Q,17,3] xy-pixel-conf)

Supports two output formats:
  - YOLO26 (end2end=True): raw model outputs [B, Q, 4+1+17*3] — boxes already decoded
    to xyxy, top-Q detections pre-selected. No further filtering needed.
  - YOLOv8 (classic):  raw model outputs [B, 4+1+17*3, num_anchors] — boxes encoded
    as cx/cy/w/h, all anchors returned. Decoded and topk-filtered here.

The format is detected automatically from the output shape.
"""

import torch
from torch import Tensor, nn


class YOLOPose(nn.Module):
    """Wraps a YOLO pose model to match the BlazeFace decoded output schema.

    Output contract:
        boxes      [Q, 4]     xyxy absolute pixel coords
        scores     [Q]        confidence in [0, 1]
        keypoints  [Q, 17, 3] (x, y, visibility) in absolute pixel coords
    """

    def __init__(self, yolo_inner_model: nn.Module, topk: int | None = 300) -> None:
        super().__init__()
        self.inner = yolo_inner_model
        self.inner.eval()
        self.topk = topk

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        outputs = self.inner(x)
        preds = outputs[0]  # [B, ?, ?]

        # ------------------------------------------------------------------
        # YOLO26 end-to-end format: [B, Q, 57]
        #   preds[b, q, 0:4]  = xyxy boxes (already decoded, absolute pixels)
        #   preds[b, q, 4]    = confidence score
        #   preds[b, q, 5]    = class_id
        #   preds[b, q, 6:57] = 51 keypoint values → [17, 3]
        # ------------------------------------------------------------------
        if preds.ndim == 3:
            preds = preds.squeeze(0)  # [Q, 57]
            decoded_boxes = preds[:, :4]               # [Q, 4]
            scores = preds[:, 4]                       # [Q]
            decoded_keypoints = preds[:, 6:].reshape(-1, 17, 3)  # [Q, 17, 3]
            return decoded_boxes, scores, decoded_keypoints

        # ------------------------------------------------------------------
        # YOLOv8 classic format: [B, 56, num_anchors]
        #   preds[b, 0:4, a] = cx, cy, w, h
        #   preds[b, 4,   a] = confidence score (sigmoid-activated)
        #   preds[b, 5:,  a] = 51 keypoint values → [17, 3]
        # ------------------------------------------------------------------
        preds = preds.squeeze(0)         # [56, num_anchors]
        num_anchors = preds.shape[1]

        cx, cy, w, h = preds[0], preds[1], preds[2], preds[3]
        decoded_boxes = torch.stack(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dim=-1
        )  # [num_anchors, 4]

        scores = preds[4]  # [num_anchors]

        kps_raw = preds[5:].permute(1, 0)           # [num_anchors, 51]
        decoded_keypoints = kps_raw.reshape(num_anchors, 17, 3)

        if self.topk is not None and self.topk > 0:
            topk_scores, topk_indices = torch.topk(
                scores, self.topk, dim=0, largest=True, sorted=True
            )
            decoded_boxes = decoded_boxes[topk_indices]
            decoded_keypoints = decoded_keypoints[topk_indices]
            scores = topk_scores

        return decoded_boxes, scores, decoded_keypoints
