"""PP-OCRv6 small — fused bucketed exporter, harness-facing.

Builds ONE fused .pte (detect + recognize) of static per-size methods plus an
`is_bucketed()` constant, for a given backend. Adapted from the standalone
`export_ppocrv6.py`; the argparse / config-emission shell is gone — the harness
authors config.json and calls `build_and_lower(backend, out_path)`.

  det (DBNet)  [1,3,S,S] f32 [0,1] RGB  -> [1,1,S,S] f32 prob   (ImageNet norm baked)
  rec (SVTR)   [1,3,48,W] f32 RGB       -> [1,W/8,18710] f32 probs

Backend fixes precision + delegate:
  xnnpack : det int8 (real-page calib) + rec fp32 (int8 rec strands dequant).
  coreml  : weight-only int8, fp16 compute (ANE).
  vulkan  : MIXED — det Vulkan fp16 + rec XNNPACK fp32 (SVTR crashes Vulkan vocab head).
"""
import json
import os

import numpy as np
import torch
from torch.export import export as te, Dim

from .wrappers import Stage, StageFp16, Norm
from .lowering import lower_save as _lower_save
from .backends import xnnpack, coreml, vulkan

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.path.join(HERE, "calib_data")          # vendored real-image calibration set
DET_ID = "PaddlePaddle/PP-OCRv6_small_det_safetensors"
REC_ID = "PaddlePaddle/PP-OCRv6_small_rec_safetensors"

DET_SQUARE = [640, 960, 1280]
DET_PORTRAIT = [(1280, 640)]                    # (H, W)
REC_WIDTHS = [160, 320, 480, 640, 1280]         # 1280 = long-line bucket
REC_H = 48
DET_DIV, REC_DIV = 32, 8                         # spatial divisors (DBNet /32, SVTR width /8)


def load_det():
    from transformers import PPOCRV6SmallDetForObjectDetection
    det = PPOCRV6SmallDetForObjectDetection.from_pretrained(DET_ID).eval()
    for mod in det.modules():                    # nearest->bilinear: 1 blob on XNNPACK
        if hasattr(mod, "interpolate_mode"):
            mod.interpolate_mode = "bilinear"
    return det


def load_rec():
    from transformers import PPOCRV6SmallRecForTextRecognition
    return PPOCRV6SmallRecForTextRecognition.from_pretrained(REC_ID).eval()


# ── real-crop calibration (PT2E int8) ──
def load_calib(widths):
    """Real test crops fitted into every bucket width (every bucket gets the full set)."""
    import cv2
    refs = json.load(open(os.path.join(TS, "refs.json")))
    imgs = [cv2.cvtColor(cv2.imread(os.path.join(TS, rel)), cv2.COLOR_BGR2RGB) for rel in refs]
    crops = {w: [] for w in widths}
    for rgb in imgs:
        h, w = rgb.shape[:2]
        nw = max(REC_DIV, int(round(REC_H * w / h)))
        for b in widths:
            rw = min(nw, b)
            r = (cv2.resize(rgb, (rw, REC_H)).astype(np.float32) / 255.0 - 0.5) / 0.5
            canvas = np.zeros((REC_H, b, 3), np.float32); canvas[:, :rw] = r
            crops[b].append(torch.from_numpy(canvas.transpose(2, 0, 1)[None]).contiguous())
    return crops


def det_page_calib(h, w, n=12):
    """Real-PAGE calibration for the detector int8 (matches the detector's [0,1] input domain)."""
    import cv2
    refs = json.load(open(os.path.join(TS, "refs.json")))
    out = []
    for rel in list(refs)[:n]:
        img = cv2.imread(os.path.join(TS, rel))
        if img is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(torch.from_numpy((rgb.transpose(2, 0, 1)[None]).copy()))   # [0,1] — norm baked
    return out or [torch.rand(1, 3, h, w)]


# ── per-stage export-program builders ──
def _wrap(model, precision):
    return StageFp16(model) if precision == "fp16" else Stage(model)


def build_eps_static(model, precision, shapes, name_of):
    mod = _wrap(model, precision)
    return {name_of(h, w): te(mod, (torch.randn(1, 3, h, w),), strict=False)
            for (h, w) in shapes}


def export_xnnpack_int8_static(load_fn, shapes_named, is_rec):
    """PT2E static int8 on XNNPACK, calibrated on real crops/pages. Returns {name: ep}."""
    calib = load_calib(REC_WIDTHS) if is_rec else None
    eps = {}
    for (h, w), mname in shapes_named:
        calib_inputs = list(calib[w]) if (is_rec and w in calib) else det_page_calib(h, w)
        eps[mname] = xnnpack.pt2e_int8(
            lambda: Stage(load_fn()), (torch.randn(1, 3, h, w),), calib_inputs,
            is_per_channel=True)
    return eps


def build_stage(stage, backend, precision):
    """Static bucketed eps for one stage ('det' or 'rec') on `backend`."""
    is_rec = stage == "rec"
    if stage == "det":
        # DBNet wants standard ImageNet on RGB (baked) -> runtime feeds raw [0,1] RGB; same
        # norm as CRAFT, so the detector input contract unifies across both OCRs.
        loader = lambda: Norm(load_det(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        shapes = [(s, s) for s in DET_SQUARE] + [(h, w) for (h, w) in DET_PORTRAIT]
        name_static = lambda h, w: (f"detect_{w}" if h == w else f"detect_{w}x{h}")
    else:
        loader = load_rec
        shapes = [(REC_H, w) for w in REC_WIDTHS]
        name_static = lambda h, w: f"recognize_{w}"

    if backend == "xnnpack" and precision == "int8":
        return export_xnnpack_int8_static(loader, [(s, name_static(*s)) for s in shapes], is_rec)
    return build_eps_static(loader(), precision, shapes, name_static)


def bucket_metadata():
    """Constant is_bucketed() -> int32[2,N]: row0 det square sizes, row1 rec widths (pad -1)."""
    rows = [sorted(DET_SQUARE), sorted(REC_WIDTHS)]
    n = max(len(r) for r in rows)
    rows = [r + [-1] * (n - len(r)) for r in rows]
    return {"is_bucketed": torch.tensor(rows, dtype=torch.int32)}


def build_and_lower(backend, out_path):
    """Build the fused bucketed PTE for `backend` and write it to absolute `out_path`."""
    assert os.path.isabs(out_path), "out_path must be absolute"
    consts = bucket_metadata()

    if backend == "vulkan":
        # MIXED-delegate: det=Vulkan fp16, rec=XNNPACK fp32 (SVTR vocab head crashes Vulkan).
        det_eps = build_stage("det", "vulkan", "fp32")
        rec_eps = build_stage("rec", "xnnpack", "fp32")
        eps = {**det_eps, **rec_eps}
        parts = {n: [vulkan.partitioner(dynamic=False, texture_limits=(2048, 2048, 2048))]
                 for n in det_eps}
        parts.update({n: [xnnpack.partitioner()] for n in rec_eps})
        return _lower_save(eps, parts, out_path,
                           transform_passes=[xnnpack.InsertContig()],
                           extra_constants=consts, width=18)

    if backend == "xnnpack":
        det_eps = build_stage("det", "xnnpack", "int8")     # detector int8 (real-page calib)
        rec_eps = build_stage("rec", "xnnpack", "fp32")     # rec fp32 (int8 strands dequant)
        eps = {**det_eps, **rec_eps}
        parts = [xnnpack.partitioner()]
        return _lower_save(eps, parts, out_path,
                           transform_passes=[xnnpack.InsertContig()],
                           extra_constants=consts, width=18)

    if backend == "coreml":
        det_eps = build_stage("det", "coreml", "int8")
        rec_eps = build_stage("rec", "coreml", "int8")
        eps = {**det_eps, **rec_eps}
        # weight-only int8, fp16 compute (ANE); static methods -> full-graph per method.
        parts = {n: [coreml.partitioner("fp16", quantize="int8", lower_full_graph=True)]
                 for n in eps}
        return _lower_save(eps, parts, out_path, extra_constants=consts, width=18)

    raise ValueError(f"unknown backend '{backend}'")
