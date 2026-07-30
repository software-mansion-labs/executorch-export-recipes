"""FastSAM export wrapper.

Converts FastSAM's raw YOLO-style output into the standard instance
segmentation contract expected by react-native-executorch:
  bbox        [B, K, 4]    x1y1x2y2 in model input pixel space
  scores      [B, K, 2]    (confidence, 0)  — class-agnostic, always class 0
  mask_logits [B, K, H, W] pre-sigmoid logits

K = topk (default 300). FastSAM does not apply topk internally (unlike YOLO),
so without pre-filtering the bmm would run on ~21k anchors at 1024x1024 —
roughly 72× more FLOPs than necessary. The topk step here keeps shapes static
(required for ExecuTorch export) while cutting bmm cost dramatically.

Usage:
    from fastsam_wrapper import FastSAMInstanceSegmentation
    from ultralytics import FastSAM as UltralyticsFastSAM

    inner = UltralyticsFastSAM("FastSAM-x.pt").model
    model = FastSAMInstanceSegmentation(inner, method_size=1024)
    # then export with torch.export / ExecuTorch pipeline
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FastSAMInstanceSegmentation(nn.Module):
    """Wraps a FastSAM backbone for ExecuTorch export.

    FastSAM (with export=True) outputs:
      pred   [B, 4+1+nm, Q]  — raw anchors: xywh, conf, nm mask coefficients
      protos [B, nm, H, W]   — prototype masks

    This wrapper applies a topk confidence pre-filter before the bmm so the
    exported graph stays fast on device. Q at 1024×1024 is ~21k; without topk
    the bmm is [1,21k,32]@[1,32,65k] ≈ 1.4B FLOPs vs [1,300,32]@[1,32,65k]
    ≈ 19M FLOPs with topk=300.
    """

    def __init__(
        self,
        inner: nn.Module,
        method_size: int,
        topk: int = 300,
        mask_size: int | None = None,
        sort_topk: bool = False,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.inner.eval()
        self.inner.model[-1].export = True
        self.sort_topk = sort_topk

        with torch.no_grad():
            preds, protos = self.inner(torch.randn(1, 3, method_size, method_size))

        # preds: [B, 4+1+nm, Q]
        Q = preds.shape[2]
        self.topk = min(topk, Q)
        self.nm: int = protos.shape[1]
        self.mask_size = mask_size
        proto_h, proto_w = protos.shape[2], protos.shape[3]
        effective = (
            f"{mask_size}x{mask_size}"
            if mask_size
            else f"{proto_h}x{proto_w} (native)"
        )
        print(
            f"  preds {preds.shape}, protos {protos.shape}, "
            f"nm={self.nm}, topk={self.topk}/{Q}, mask_size={effective}"
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        preds, protos = self.inner(x)   # preds: [B, 4+1+nm, Q]

        # Select top-k anchors by confidence before the expensive bmm.
        # topk is fixed at init time so shapes remain static for export.
        conf_scores = preds[:, 4, :]                          # [B, Q]
        top_idx = conf_scores.topk(self.topk, dim=1, sorted=self.sort_topk).indices  # [B, K]
        preds = preds.gather(
            2, top_idx.unsqueeze(1).expand(-1, preds.shape[1], -1)
        )                                                      # [B, 37, K]

        preds = preds.transpose(1, 2)                          # [B, K, 37]

        # xywh -> xyxy
        cx, cy, w, h = preds[..., 0], preds[..., 1], preds[..., 2], preds[..., 3]
        bbox = torch.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
        )  # [B, K, 4]

        # FastSAM is class-agnostic: confidence + constant class index 0
        conf = preds[..., 4]                                   # [B, K]
        classes = torch.zeros_like(conf)                       # [B, K]

        coeffs = preds[..., 5:]                                # [B, K, nm]

        if self.mask_size is not None:
            protos = F.interpolate(
                protos,
                size=(self.mask_size, self.mask_size),
                mode="bilinear",
                align_corners=False,
            )

        B, C, H, W = protos.shape
        mask_logits = torch.bmm(coeffs, protos.reshape(B, C, H * W)).reshape(
            B, -1, H, W
        )  # [B, K, H, W]

        return bbox[0], conf[0], classes[0], mask_logits[0]
