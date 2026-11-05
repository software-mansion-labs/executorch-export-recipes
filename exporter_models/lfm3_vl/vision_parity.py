"""Compare the exportable vision encoder with upstream `get_image_features`.

Feeds one 512x512 image through the HF processor (no resize: the RNE runner
letterboxes to 512x512 before the encoder) and through `VisionEncoder`, both
in fp32, and reports the cosine / max error of the projected image tokens.

    python exporter_models/lfm3_vl/vision_parity.py --source <LFM3-VL-2B snapshot> [--image img.jpg]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lfm3.vision import VisionConfig, VisionEncoder, convert  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--image")
    args = ap.parse_args()

    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if args.image:
        img = Image.open(args.image).convert("RGB").resize((512, 512), Image.BILINEAR)
    else:
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:512, 0:512]
        arr = np.stack([xx % 256, yy % 256, (xx + yy) % 256], -1) + rng.integers(0, 32, (512, 512, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    proc = AutoProcessor.from_pretrained(args.source, trust_remote_code=True)
    inputs = proc.image_processor(images=[img], do_resize=False, return_tensors="pt")
    hf = AutoModelForImageTextToText.from_pretrained(
        args.source, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        want = hf.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
        ).pooler_output[0]
    del hf

    hf_cfg = json.loads((Path(args.source) / "config.json").read_text())
    enc = VisionEncoder(hf_cfg, (512, 512))
    state = convert(load_file(str(Path(args.source) / "model.safetensors")), VisionConfig.from_hf(hf_cfg))
    missing, unexpected = enc.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    raw = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().contiguous()[None]
    with torch.no_grad():
        got = enc.eval()(raw)[0]
    cos = torch.nn.functional.cosine_similarity(got, want, dim=-1)
    print(f"tokens {tuple(got.shape)} vs {tuple(want.shape)}")
    print(f"per-token cos min {cos.min().item():.6f} mean {cos.mean().item():.6f}; "
          f"max abs err {(got - want).abs().max().item():.4g} (|ref| max {want.abs().max().item():.4g})")


if __name__ == "__main__":
    main()
