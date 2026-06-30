"""Real-image calibration loaders for PT2E int8 (PaddleOCR exporters).

int8 MUST be calibrated on real activations — random noise is out-of-distribution
and makes int8 look broken (see int8-calibration-real-data). The shared
calibration set lives next to the EasyOCR weights.
"""
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIB_DIR = os.path.join(HERE, "..", "ocr", "export", "calibration_images")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_images(hw, *, normalize="imagenet", n=8, calib_dir=None):
    """Real images resized to `hw`=(H,W), as contiguous NCHW fp32 tensors.

    normalize:
      "imagenet"   -> (x/255 - mean) / std   (PP-LCNet, SLANet backbone)
      "div255"     -> x/255                   (UVDoc; fused helpers)
      (mean, std)  -> custom per-channel normalization
    Pads out to `n` by re-using images when the directory has fewer.
    """
    import cv2
    h, w = hw
    calib_dir = calib_dir or DEFAULT_CALIB_DIR
    out = []
    for fn in sorted(os.listdir(calib_dir)):
        img = cv2.imread(os.path.join(calib_dir, fn), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if normalize == "imagenet":
            rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        elif isinstance(normalize, tuple):
            mean, std = normalize
            rgb = (rgb - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
        # "div255" -> no further normalization
        out.append(torch.from_numpy(rgb.transpose(2, 0, 1)[None]).contiguous())
    while out and len(out) < n:
        out.append(out[len(out) % len(out)])
    return out
