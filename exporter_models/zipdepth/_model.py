"""Pinned ZipDepth base NPU checkpoint; raw float depth, RGB input in [0,1]."""

import hashlib
import sys
from pathlib import Path

import torch

REVISION = "91f3fd21e131641f51e8d35736d1958350180e3a"
CHECKPOINT_SHA256 = "627c04fda584133ead4310074884a4a037061b4c01ba86e73e492ea30fab570d"
CHECKPOINT_FILE = f"zipdepth_base_npu_{REVISION}.pth"
CHECKPOINT_URL = f"https://raw.githubusercontent.com/fabiotosi92/ZipDepth/{REVISION}/checkpoints/zipdepth_base_npu.pth"


class ZipDepthModel(torch.nn.Module):
    def __init__(self, fuse=True):
        super().__init__()
        sys.path.insert(0, str(Path(__file__).parent / "_upstream"))
        from zipdepth.model.architecture import create_model
        from zipdepth.utils.model_utils import (
            fuse_remaining_conv_bn,
            strip_state_dict_prefixes,
        )

        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / CHECKPOINT_FILE
        if not checkpoint.exists():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(CHECKPOINT_URL, str(checkpoint))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = strip_state_dict_prefixes(state.get("model_state_dict", state))
        self.model = create_model(
            variant="base", global_mode="balanced", upsample_unfold=False
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        if fuse:
            self.model.fuse_for_inference()
            fuse_remaining_conv_bn(self.model)

    def forward(self, pixel_values):
        # Upstream forward contains ImageNet normalization: callers supply RGB / 255 only.
        return self.model(pixel_values).squeeze(1)


def build_model(fuse=True):
    return ZipDepthModel(fuse=fuse).eval()
