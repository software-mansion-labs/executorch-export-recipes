"""The exported graph: the face-mesh backbone plus its output reshaping.

Upstream returns `(1x1404x1x1 landmarks, 1x1x1x1 logit)` and leaves the caller
to flatten and to sigmoid. Both are baked in here so the `.pte` publishes the
shape the runtime consumes, `(468x3, 1)`, with the score already in [0, 1].

Landmark coordinates stay in input-pixel units (0..192 for x and y, the same
scale for z), which is what the JS side scales back to the source image.
"""

import torch
import torch.nn as nn

from .facial_lm_model import NUM_LANDMARKS, FacialLM_Model
from .weights import checkpoint_path


class FaceMesh(nn.Module):
    def __init__(self, model: FacialLM_Model):
        super().__init__()
        self.model = model

    def forward(self, x):
        landmarks, logit = self.model(x)
        return (
            landmarks.reshape(NUM_LANDMARKS, 3),
            torch.sigmoid(logit).reshape(1),
        )


def build() -> FaceMesh:
    """Load the published checkpoint into the eval-mode exported wrapper."""
    model = FacialLM_Model()
    model.load_state_dict(torch.load(str(checkpoint_path()), map_location="cpu"))
    return FaceMesh(model.eval()).eval()
