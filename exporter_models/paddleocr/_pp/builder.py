"""PP-OCRv6 small — fused DYNAMIC exporter, harness-facing.

Builds ONE fused .pte per backend with true-dynamic `detect` + `recognize` methods
plus `get_dynamic_dims_<method>` companions — the runtime validation contract: a
no-arg method returning one int32 [rank, 3] tensor per tensor input of the target
method, each row [min, max, step] (static dims encoded [n, n, 1]). CoreML has no
RangeDim, so it ships enumerated single methods + `<method>_enum_shapes` constants
instead of companions (the runtime snaps to the nearest enumerated shape).

  det (DBNet)  [1,3,H,W] f32 [0,1] RGB -> [1,1,H,W] f32 prob   (ImageNet norm baked)
               H, W dynamic in [640, 1280] step 32
  rec (SVTR)   [1,3,48,W] f32 RGB      -> [1,W/8,18710] f32 probs
               W dynamic in [160, 1280] step 8 (H fixed 48)

Backend fixes precision + delegate:
  xnnpack : det + rec fp32 dynamic. int8 is OFF THE TABLE under dynamic shapes:
            static-activation quant params don't transfer across scales (det IoU~0
            even at the calib shape) and int8 rec wrecks the attention SVTR anyway.
  coreml  : ENUMERATED — weight-only int8, fp16 compute (ANE), enum_shapes baked.
  vulkan  : MIXED — det Vulkan fp16 dynamic via resize + rec XNNPACK fp32 dynamic
            (SVTR crashes the Vulkan vocab head on device: make_ivec2).
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

# Dynamic bounds (legacy-proven). CoreML enumerates concrete shapes from the same sets.
DET_MIN, DET_MAX = 640, 1280
REC_W_MIN, REC_W_MAX = 160, 1280
DET_SQUARE = [640, 960, 1280]                   # CoreML enumerated det shapes (square)
DET_PORTRAIT = [(1280, 640)]                    # (H, W) CoreML enumerated portrait
REC_WIDTHS = [160, 320, 480, 640, 1280]         # CoreML enumerated + calibration widths
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


def _det_loader():
    # DBNet wants standard ImageNet on RGB (baked) -> runtime feeds raw [0,1] RGB; same
    # norm as CRAFT, so the detector input contract unifies across both OCRs.
    return Norm(load_det(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# ── real-crop calibration (PT2E int8) ──
def load_calib(widths):
    """Real test crops fitted into every calibration width (each gets the full set)."""
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


# ── dynamic dims + companions ──
def _dyn(stage):
    """(dynamic_shapes dict, example (H,W)) for one stage's symbolic export."""
    if stage == "det":
        h = Dim("det_h", min=DET_MIN // DET_DIV, max=DET_MAX // DET_DIV)
        w = Dim("det_w", min=DET_MIN // DET_DIV, max=DET_MAX // DET_DIV)
        return {2: DET_DIV * h, 3: DET_DIV * w}, (DET_MAX, DET_MAX)
    w = Dim("rec_w", min=REC_W_MIN // REC_DIV, max=REC_W_MAX // REC_DIV)
    return {3: REC_DIV * w}, (REC_H, REC_W_MAX)


def companions():
    """get_dynamic_dims_<method> programs: int32 [rank,3] rows [min,max,step] per input."""
    from exporter.dynamic_dims import build_dynamic_dims_program
    det = [[1, 1, 1], [3, 3, 1],
           [DET_MIN, DET_MAX, DET_DIV], [DET_MIN, DET_MAX, DET_DIV]]
    rec = [[1, 1, 1], [3, 3, 1], [REC_H, REC_H, 1],
           [REC_W_MIN, REC_W_MAX, REC_DIV]]
    return {"get_dynamic_dims_detect": build_dynamic_dims_program([det]),
            "get_dynamic_dims_recognize": build_dynamic_dims_program([rec])}


# ── per-stage export-program builders ──
def _wrap(model, precision):
    return StageFp16(model) if precision == "fp16" else Stage(model)


def build_stage_dynamic(stage, precision):
    """ONE symbolic-dim ep for `stage` ('det'/'rec') at `precision` ('fp32'/'fp16'/'int8').

    int8 (det only) = PT2E static int8 with dynamic shapes, real-page calibrated at the
    max size. SVTR (rec) has no recurrence so it exports symbolically at any precision,
    but int8 is never used for it (see module docstring).
    """
    dims, example = _dyn(stage)
    loader = _det_loader if stage == "det" else load_rec
    if precision == "int8":
        h, w = example
        return xnnpack.pt2e_int8(
            lambda: Stage(loader()), (torch.randn(1, 3, h, w),), det_page_calib(h, w),
            is_per_channel=True, dynamic_shapes=dims)
    return te(_wrap(loader(), precision), (torch.randn(1, 3, *example),),
              dynamic_shapes=({**dims},), strict=False)


def build_stage_enumerated(stage, precision):
    """(ep, enum shape list) for CoreML: symbolic export + concrete enumerated shapes."""
    dims, example = _dyn(stage)
    loader = _det_loader if stage == "det" else load_rec
    ep = te(_wrap(loader(), precision), (torch.randn(1, 3, *example),),
            dynamic_shapes=({**dims},), strict=False)
    if stage == "det":
        shapes = [(s, s) for s in DET_SQUARE] + list(DET_PORTRAIT)
    else:
        shapes = [(REC_H, w) for w in REC_WIDTHS]
    return ep, [[1, 3, h, w] for (h, w) in shapes]


def build_and_lower(backend, out_path):
    """Build the fused dynamic PTE for `backend` and write it to absolute `out_path`."""
    assert os.path.isabs(out_path), "out_path must be absolute"

    if backend == "xnnpack":
        # BOTH stages fp32: static-activation int8 quant params do not transfer across
        # dynamic scales (det corr 0.59-0.88 / IoU~0 even at the calibration shape;
        # multi-size calib stays unstable) — fp32 dynamic is bit-exact at every shape.
        # fp16 is no lever here: the XNNPACK delegate blob stays fp32-sized (measured
        # 9.94 vs 9.90 MB) with worse delegation. int8 rec additionally wrecks SVTR.
        eps = {"detect": build_stage_dynamic("det", "fp32"),
               "recognize": build_stage_dynamic("rec", "fp32"),
               **companions()}
        return _lower_save(eps, [xnnpack.partitioner()], out_path,
                           transform_passes=[xnnpack.InsertContig()], width=28)

    if backend == "vulkan":
        # MIXED-delegate: det=Vulkan fp16 dynamic via resize, rec=XNNPACK fp32 dynamic.
        # Delegate the DBNet FPN bilinear upsamples to Vulkan (else portable upsample
        # splits detect into 4 blobs -> GPU<->CPU round-trips; scale_factor path only,
        # safe on Xclipse). fp16 = fp32 graph + force_fp16 (never a .half() graph).
        vulkan.patch_upsample_resize()
        eps = {"detect": build_stage_dynamic("det", "fp32"),
               "recognize": build_stage_dynamic("rec", "fp32"),
               **companions()}
        parts = {"detect": [vulkan.partitioner(dynamic=True,
                                               texture_limits=(2048, 2048, 2048))]}
        parts.update({n: [xnnpack.partitioner()] for n in eps if n != "detect"})
        return _lower_save(eps, parts, out_path,
                           transform_passes=[xnnpack.InsertContig()], width=28)

    if backend == "coreml":
        # weight-only int8, fp16 compute (ANE); enumerated -> full-graph per method,
        # `<method>_enum_shapes` constants for the runtime snap (no companions).
        det_ep, det_shapes = build_stage_enumerated("det", "int8")
        rec_ep, rec_shapes = build_stage_enumerated("rec", "int8")
        eps = {"detect": det_ep, "recognize": rec_ep}
        enum_map = {n: {ep.graph_signature.user_inputs[0]: shapes}
                    for n, ep, shapes in [("detect", det_ep, det_shapes),
                                          ("recognize", rec_ep, rec_shapes)]}
        parts = {n: [coreml.partitioner("fp16", quantize="int8", lower_full_graph=True,
                                        extra_specs=[coreml.enumerated_spec(eps[n], enum_map[n])])]
                 for n in eps}
        return _lower_save(eps, parts, out_path, enum_map=enum_map, width=28)

    raise ValueError(f"unknown backend '{backend}'")
