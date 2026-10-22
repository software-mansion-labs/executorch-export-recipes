"""The exported graph: the face-mesh backbone plus its output decoding.

Upstream returns `(1x1404x1x1 landmarks, 1x1x1x1 logit)` and leaves the caller
to flatten, to sigmoid and to work out where the face is. All of that is baked
in here so the `.pte` publishes the same `(boxes, scores, keypoints)` contract
as every other keypoint model, with one candidate:

- `boxes[1, 4]`    the xyxy hull of the mesh, in input-pixel units;
- `scores[1]`      the face-presence probability;
- `keypoints[1, 468, 4]` per landmark: x, y, confidence, z.

The mesh carries no per-landmark confidence, so slot 2 is a constant 1 — the
same thing blazeface's export does with its six landmarks. Slot 3 is upstream's
relative depth, on the same scale as x.
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
        xyz = landmarks.reshape(1, NUM_LANDMARKS, 3)
        xy = xyz[..., :2]

        boxes = torch.cat([xy.amin(dim=1), xy.amax(dim=1)], dim=-1)
        scores = torch.sigmoid(logit).reshape(1)
        # Concatenated slices rather than a stack of selects: the selects each
        # become a `select_copy` the partitioners leave on the CPU. The ones
        # stay an op rather than a registered buffer, which the int8 quantizer
        # gives a degenerate (min == max) scale and XNNPACK then rejects.
        keypoints = torch.cat([xy, torch.ones_like(xy[..., :1]), xyz[..., 2:]], dim=-1)
        return boxes, scores, keypoints


def build() -> FaceMesh:
    """Load the published checkpoint into the eval-mode exported wrapper."""
    model = FacialLM_Model()
    model.load_state_dict(torch.load(str(checkpoint_path()), map_location="cpu"))
    return FaceMesh(model.eval()).eval()
