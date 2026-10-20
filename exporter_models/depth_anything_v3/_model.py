"""DA3 Small, single-view raw depth only, with an unchanged upstream backbone/head."""

import sys
from pathlib import Path
import torch

SOURCE = "depth-anything/DA3-SMALL"
REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"
UPSTREAM_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"


class DepthAnythingV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(Path(__file__).parent / "_upstream"))
        from depth_anything_3.model.dinov2.dinov2 import DinoV2
        from depth_anything_3.model.dualdpt import DualDPT
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        self.backbone = DinoV2(
            name="vits",
            out_layers=[5, 7, 9, 11],
            alt_start=4,
            qknorm_start=4,
            rope_start=4,
            cat_token=True,
        )
        self.head = DualDPT(
            dim_in=768, output_dim=2, features=64, out_channels=[48, 96, 192, 384]
        )
        weights = load_file(
            hf_hub_download(SOURCE, "model.safetensors", revision=REVISION)
        )
        selected = {
            k.removeprefix("model."): v
            for k, v in weights.items()
            if k.startswith(("model.backbone.", "model.head."))
        }
        # Safetensors stores the auxiliary projections' shared LayerNorm once.
        # Restore its aliases so strict loading still catches unrelated mismatches.
        for level in (1, 2, 3):
            for field in ("weight", "bias"):
                selected[f"head.scratch.output_conv2_aux.{level}.2.{field}"] = selected[
                    f"head.scratch.output_conv2_aux.0.2.{field}"
                ]
        self.load_state_dict(selected, strict=True)

    def forward(self, pixel_values):
        features, _ = self.backbone(
            pixel_values.unsqueeze(1), ref_view_strategy="first"
        )
        return self.head(
            features, pixel_values.shape[-2], pixel_values.shape[-1], patch_start_idx=0
        )["depth"][:, 0]


def build_model():
    return DepthAnythingV3().eval()
