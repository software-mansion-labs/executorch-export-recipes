"""A/B the PP-OCRv6 detector's int8 calibration band against eager fp32.

The shipped XNNPACK build is fp32 because the int8 attempt measured
`corr 0.59-0.88 / IoU~0 even at the calibration shape`. That attempt calibrated
at DET_MAX (1280x1280) only, which is the size where static-activation int8 is
known broken for this model family. EasyOCR hit the same wall and solved it by
calibrating one scale set over a band of SMALL sizes and declaring the band.

This measures quantization error directly on the converted graph (no ExecuTorch
runtime involved), so a band that loses accuracy is rejected before anything is
exported.

    uv run --project exporter_models/paddleocr python quant_experiment.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _pp import builder as B  # noqa: E402
from _pp.backends import xnnpack  # noqa: E402
from _pp.wrappers import Stage  # noqa: E402

EVAL_SIZES = [(640, 640), (800, 800), (960, 960), (1280, 1280), (1280, 640)]

# Calibration bands to compare. "max" is what ships today.
BANDS = {
    "max_1280_only": [(1280, 1280)],
    "narrow_640_800": [(640, 640), (704, 704), (768, 768), (800, 800)],
    "narrow_plus_aspect": [(640, 640), (768, 768), (800, 800), (1280, 640), (640, 1280)],
}


def calib_for(sizes, per_size=6):
    out = []
    for (h, w) in sizes:
        out.extend(B.det_page_calib(h, w, n=per_size))
    return out


def eval_pages(h, w, n=4):
    """Held-out-ish eval pages: take the TAIL of refs so they differ from calib's head."""
    import cv2
    refs = list(json.load(open(os.path.join(B.TS, "refs.json"))))[-n:]
    mean = np.array(B.IMAGENET_MEAN, np.float32).reshape(1, 1, 3)
    std = np.array(B.IMAGENET_STD, np.float32).reshape(1, 1, 3)
    out = []
    for rel in refs:
        img = cv2.imread(os.path.join(B.TS, rel))
        if img is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - mean) / std
        out.append(torch.from_numpy(rgb.transpose(2, 0, 1)[None].copy()))
    return out


def corr(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def iou(a, b, thr=0.3):
    am, bm = a > thr, b > thr
    union = np.logical_or(am, bm).sum()
    return float(np.logical_and(am, bm).sum() / union) if union else 1.0


def main():
    torch.manual_seed(0)
    dims, _ = B._dyn("det")

    print("building eager fp32 reference ...", flush=True)
    ref = Stage(B._det_loader()).eval()

    refs_out = {}
    with torch.no_grad():
        for (h, w) in EVAL_SIZES:
            refs_out[(h, w)] = [ref(x).numpy() for x in eval_pages(h, w)]

    results = {}
    for name, sizes in BANDS.items():
        print(f"\n=== calibration band: {name} ({sizes}) ===", flush=True)
        calib = calib_for(sizes)
        # Trace/prepare/convert at the band's largest size so the example shape is in-band.
        ex_h, ex_w = max(sizes, key=lambda s: s[0] * s[1])
        gm = xnnpack.pt2e_int8(
            lambda: Stage(B._det_loader()), (torch.randn(1, 3, ex_h, ex_w),), calib,
            is_per_channel=True, dynamic_shapes=dims)
        gm = gm.module() if hasattr(gm, "module") else gm

        row = {}
        with torch.no_grad():
            for (h, w) in EVAL_SIZES:
                cs, is_ = [], []
                for x, r in zip(eval_pages(h, w), refs_out[(h, w)]):
                    q = gm(x)
                    q = (q[0] if isinstance(q, (tuple, list)) else q).numpy()
                    cs.append(corr(r, q))
                    is_.append(iou(r, q))
                row[(h, w)] = (float(np.mean(cs)), float(np.mean(is_)))
                print(f"  {h}x{w}: corr {row[(h,w)][0]:.4f}  IoU@0.3 {row[(h,w)][1]:.3f}",
                      flush=True)
        results[name] = row

    print("\n\n=== summary (corr / IoU@0.3) ===")
    hdr = "band".ljust(20) + "".join(f"{h}x{w}".rjust(18) for (h, w) in EVAL_SIZES)
    print(hdr)
    for name, row in results.items():
        line = name.ljust(20)
        for k in EVAL_SIZES:
            c, i = row[k]
            line += f"{c:.4f}/{i:.3f}".rjust(18)
        print(line)


if __name__ == "__main__":
    main()
