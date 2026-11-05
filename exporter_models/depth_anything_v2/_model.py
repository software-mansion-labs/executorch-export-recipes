"""Pinned Depth Anything V2 Small; raw relative depth, no display normalization."""
import torch

SOURCE = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


class DepthAnythingV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForDepthEstimation

        self.model = AutoModelForDepthEstimation.from_pretrained(
            SOURCE, revision=REVISION, attn_implementation="eager",
        ).eval()

    def forward(self, pixel_values):
        return self.model(pixel_values, return_dict=False)[0]


def build_model():
    return DepthAnythingV2().eval()
