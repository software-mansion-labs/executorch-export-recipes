"""MediaPipe face-landmark ("face mesh") model in PyTorch.

Vendored from https://github.com/tiqq111/mediapipe_pytorch (facial_landmarks/
facial_lm_model.py). The upstream `forward` carries `torch.no_grad` and the
numpy-fed `predict` / `batch_predict` / `test` helpers; both are dropped here
because `torch.export` traces the module directly, and the input's
ReflectionPad2d is spelled as slice-concats (see `reflect_pad_top_left`).

The two heads are named the wrong way round upstream (`self.confidence`
regresses the landmarks and `self.facial_landmarks` scores them). The names are
kept as-is so the published state dict loads unchanged.
"""

import torch.nn as nn

from .utils import FacialLMBasicBlock, reflect_pad_top_left

#: Landmarks the model regresses, 3 channels (x, y, z) each: 468 * 3 = 1404.
NUM_LANDMARKS = 468


class FacialLM_Model(nn.Module):
    """Input 1x3x192x192 in [-1, 1]; outputs (1x1404x1x1, 1x1x1x1)."""

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.PReLU(16),
            FacialLMBasicBlock(16, 16),
            FacialLMBasicBlock(16, 16),
            FacialLMBasicBlock(16, 32, stride=2),
            FacialLMBasicBlock(32, 32),
            FacialLMBasicBlock(32, 32),
            FacialLMBasicBlock(32, 64, stride=2),
            FacialLMBasicBlock(64, 64),
            FacialLMBasicBlock(64, 64),
            FacialLMBasicBlock(64, 128, stride=2),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128, stride=2),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128),
        )

        # Landmark regression head, despite the name.
        self.confidence = nn.Sequential(
            FacialLMBasicBlock(128, 128, stride=2),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128),
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.PReLU(32),
            FacialLMBasicBlock(32, 32),
            nn.Conv2d(
                in_channels=32,
                out_channels=NUM_LANDMARKS * 3,
                kernel_size=3,
                stride=3,
                padding=0,
                bias=True,
            ),
        )

        # Face-presence score head, despite the name.
        self.facial_landmarks = nn.Sequential(
            FacialLMBasicBlock(128, 128, stride=2),
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.PReLU(32),
            FacialLMBasicBlock(32, 32),
            nn.Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=3,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x):
        x = reflect_pad_top_left(x)
        features = self.backbone(x)
        confidence = self.facial_landmarks(features)
        facial_landmarks = self.confidence(features)
        return [facial_landmarks, confidence]
