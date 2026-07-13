"""EasyOCR (CRAFT detector + per-language CRNN recognizer) — fused DYNAMIC exporter.

Builds ONE fused .pte per backend + language: a true-dynamic `detect` (CRAFT) with a
`get_dynamic_dims_detect` companion (no-arg method returning one int32 [rank, 3]
tensor per tensor input, rows [min, max, step]; static dims [n, n, 1]) plus a
fixed-width `recognize` (CRNN) declared via a `recognize_enum_shapes` constant —
the CRNN's BiLSTM cannot be dynamic on CPU/ANE (symbolic T -> torch.scan -> portable
~15s; concrete T -> nn.LSTM unrolls into matmuls that fully delegate), so the C++
snaps every crop to the baked width.

  CRAFT (detect)  [1,3,H,W] f32 [0,1] RGB -> ([1,H/2,W/2,2] score, [1,32,H/2,W/2] feat)
                  H,W dynamic, step 32 — per-backend band (see companion):
                  xnnpack [320, 800] both dims; vulkan H [800, 1280], W [320, 1280]
                  (ImageNet norm BAKED -> runtime feeds raw [0,1] RGB; unifies w/ DBNet)
  CRNN  (recognize) [1,3,64,512] f32 RGB  -> [1, 127, V] probs, ONE fixed width
                  (RGB->gray 1x1 conv + softmax BAKED -> same contract as SVTR)

Per-backend precision + delegate (fixed; no knob):
  xnnpack : detect NARROW-BAND static-act int8 dynamic [320,800] (qs8 chains =
            bucketed-class speed; norm folded into conv1) + recognize int8 @512.
  coreml  : ENUMERATED detect (weight-only int8, fp16 ANE) + recognize static @512;
            enum_shapes constants, no companions (no RangeDim on CoreML).
  vulkan  : MIXED — detect Vulkan fp16 dynamic via resize (fp32 graph + force_fp16;
            int8 q8ta cannot resize) + recognize int8 on XNNPACK @512.

The scan BiLSTM (arch.ScanBiL) stays in the vendored arch for a future true-dynamic
recognizer; nothing here uses it.
"""
import json
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.export import export as te, Dim

from .arch import (CRAFT, CRAFTDyn, CRNNModel, ScanBiL, copyStateDict)
from .lowering import lower_save
from .backends import xnnpack, coreml, vulkan
from . import weights

HERE = os.path.dirname(os.path.abspath(__file__))

# Dynamic bounds. Vulkan (fp16 resizes losslessly): legacy-proven 800-1280 range.
DET_H_MIN, DET_H_MAX = 800, 1280
DET_W_MIN, DET_W_MAX = 320, 1280
# XNNPACK: static-act int8 over the FULL [320, 1280] range, calibrated ONLY at
# <=800 sizes (DET_XNN_BAND). Static int8 scales are stable within a narrow size
# band (corr 0.97-0.9997 over 4 imgs at <=800) but no calibration makes them hold
# at 1280 (~0.55-0.97 content-dependent; the OG dynamic AND the old bucketed
# detect_1280 were equally broken there — measured on the shipped artifacts).
# DELIBERATE tradeoff (user call): keep 1280 in the contract as best-effort rather
# than cap the range; quality above ~800 is not guaranteed. qs8 chains = bucketed-
# class speed (S24 811ms@800² vs bucketed 731ms; qd8 alt: correct@1280 but 1296ms).
DET_XNN_MIN, DET_XNN_MAX = 320, 1280
DET_XNN_BAND = [(640, 640), (736, 736), (800, 800), (800, 320), (320, 800)]  # calib
CRAFT_SHAPES = [(800, 800), (1280, 1280), (1280, 320)]   # (H, W) CoreML enumerated set
CRNN_H = 64
CRNN_FIXED_W = 512   # recognize width, all backends (max bucket = no truncation)
CRAFT_DIV, CRNN_DIV = 32, 4                               # spatial divisors


# ── weight loaders ──
def load_craft(dynamic):
    m = (CRAFTDyn if dynamic else CRAFT)()
    m.load_state_dict(copyStateDict(torch.load(
        str(weights.craft_path()), map_location="cpu", weights_only=False)))
    return m.eval()


def load_crnn(lang):
    m = CRNNModel(input_channel=1, output_channel=256, hidden_size=256,
                  num_class=weights.num_class(lang))
    sd = torch.load(str(weights.crnn_path(lang)), map_location="cpu", weights_only=False)
    m.load_state_dict(OrderedDict((k[7:], v) for k, v in sd.items()))
    return m.eval()


# ── unified-contract bakes ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DetNorm(nn.Module):
    """Bake CRAFT's ImageNet norm INTO the graph so the runtime feeds raw [0,1] RGB
    (no client norm) — converges the detector input contract with PaddleOCR's DBNet."""
    def __init__(self, m, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return self.m((x - self.mean) / self.std)


def fold_detnorm_into_conv1(craft, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Fold (x-mean)/std into CRAFT's first conv IN PLACE — exact algebra, same [0,1]
    RGB runtime contract as DetNorm, but no sub/div ops in the graph.

    Required for the XNNPACK qd8 path: a qd8 dynamic-quantize convert whose input is
    produced INSIDE the delegate blob (here the broadcast sub/div) fails
    xnn_reshape_runtime with xnn_status_invalid_parameter on every execute; feeding
    the convert straight from the graph input sidesteps it (ET delegate bug).

      W'[o,c,:,:] = W[o,c,:,:] / std_c
      b'[o]       = b[o] - sum_c,kh,kw W[o,c,kh,kw] * mean_c / std_c
    """
    conv = craft.basenet.slice1[0]
    assert isinstance(conv, nn.Conv2d) and conv.in_channels == 3, conv
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    with torch.no_grad():
        w = conv.weight.data
        boff = (w * (m / s)).sum(dim=(1, 2, 3))
        conv.weight.data = w / s
        conv.bias = (nn.Parameter(-boff) if conv.bias is None
                     else nn.Parameter(conv.bias.data - boff))
    return craft


class UnifiedCRNN(nn.Module):
    """Bake a fixed RGB->gray 1x1 conv (cv2 weights 0.299/0.587/0.114) + output softmax, so
    the CRNN takes [1,3,H,W] RGB (same as SVTR) and emits [1,T,V] PROBS (same as SVTR).
    Lossless: the gray weights sum to 1, so they commute with the (x-0.5)/0.5 norm AND
    reproduce cv2 grayscale exactly (verified max|delta| ~= 3e-9)."""
    def __init__(self, m):
        super().__init__()
        self.crnn = m
        g = nn.Conv2d(3, 1, 1, bias=False)
        g.weight.data = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.gray = g

    def forward(self, x):
        return self.crnn(self.gray(x)).softmax(-1)


# ── calibration (PT2E int8) — real images only (random noise -> garbage on real text) ──
def crnn_calib(widths):
    import cv2
    grays = []
    for i in range(1, 5):
        g = cv2.imread(os.path.join(HERE, "calibration_images", f"{i}.png"),
                       cv2.IMREAD_GRAYSCALE)
        if g is not None:
            grays.append(g)
    out = {}
    for w in widths:
        xs = []
        for g in grays:
            band = max(8, g.shape[0] // 4)
            regions = [g] + [g[k * band:(k + 1) * band, :] for k in range(3)]
            for r in regions:
                rs = cv2.resize(r, (w, CRNN_H)).astype("float32") / 255.0
                t = torch.from_numpy((rs - 0.5) / 0.5)      # [-1,1], CRNN mean=std=0.5
                xs.append(t.view(1, 1, CRNN_H, w).repeat(1, 3, 1, 1))   # 3ch RGB (gray-conv baked)
        out[w] = xs or [torch.rand(1, 3, CRNN_H, w) for _ in range(8)]
    return out


def craft_calib(shapes):
    import cv2
    out = {}
    for (h, w) in shapes:
        xs = []
        for i in range(1, 5):
            p = os.path.join(HERE, "calibration_images", f"{i}.png")
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            r = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            xs.append(torch.from_numpy(r.transpose(2, 0, 1)[None].copy()))   # [0,1] — ImageNet norm baked
        out[(h, w)] = xs or [torch.rand(1, 3, h, w) for _ in range(8)]
    return out


# ── dynamic dims + companion ──
def _det_dyn():
    """(dynamic_shapes dict, example (H,W)) for the symbolic CRAFT export."""
    h = Dim("det_h", min=DET_H_MIN // CRAFT_DIV, max=DET_H_MAX // CRAFT_DIV)
    w = Dim("det_w", min=DET_W_MIN // CRAFT_DIV, max=DET_W_MAX // CRAFT_DIV)
    return {2: CRAFT_DIV * h, 3: CRAFT_DIV * w}, (DET_H_MAX, DET_W_MAX)


def det_companion(h_bounds=(DET_H_MIN, DET_H_MAX), w_bounds=(DET_W_MIN, DET_W_MAX)):
    """get_dynamic_dims_detect: int32 [4,3] rows [min,max,step] for the CRAFT input.
    Bounds are per-backend: XNNPACK ships the narrow [320,800] band, Vulkan the full
    [800..1280 x 320..1280] range."""
    from exporter.dynamic_dims import build_dynamic_dims_program
    rows = [[1, 1, 1], [3, 3, 1],
            [h_bounds[0], h_bounds[1], CRAFT_DIV], [w_bounds[0], w_bounds[1], CRAFT_DIV]]
    return {"get_dynamic_dims_detect": build_dynamic_dims_program([rows])}


# ── per-stage export-program builders ──
def build_detect_dynamic(precision):
    """ONE symbolic-dim `detect` ep.

    int8 (XNNPACK) = NARROW-BAND static-act int8: dims in [320, 800], calibrated on
    real pages at DET_XNN_BAND sizes, norm FOLDED into conv1 (a qd8/quantize convert
    fed by in-blob sub/div fails xnn_reshape_runtime; the fold also serves qs8 by
    keeping the input observer on the raw [0,1] image). qs8 int8-int8 chains run at
    bucketed-class speed — vs qd8 (1.6x slower) or fp32 (5x, and fp16 is fake on
    XNNPACK: fp32-sized AND fp32-speed).

    fp32 (Vulkan) = plain graph with DetNorm baked over the full [800..1280 x
    320..1280] range (fp16 via force_fp16 resizes losslessly; never .half())."""
    if precision == "int8":
        h = Dim("det_h", min=DET_XNN_MIN // CRAFT_DIV, max=DET_XNN_MAX // CRAFT_DIV)
        w = Dim("det_w", min=DET_XNN_MIN // CRAFT_DIV, max=DET_XNN_MAX // CRAFT_DIV)
        dims = {2: CRAFT_DIV * h, 3: CRAFT_DIV * w}
        cmap = craft_calib(DET_XNN_BAND)
        cal = [x for s in DET_XNN_BAND for x in cmap[s][:4]]
        return xnnpack.pt2e_int8(
            fold_detnorm_into_conv1(load_craft(dynamic=True)).eval(),
            (torch.randn(1, 3, DET_XNN_MAX, DET_XNN_MAX),), cal,
            is_per_channel=True, dynamic_shapes=dims)
    dims, example = _det_dyn()
    return te(DetNorm(load_craft(dynamic=True)), (torch.randn(1, 3, *example),),
              dynamic_shapes=({**dims},), strict=False)


def build_detect_enumerated():
    """(ep, enum shape list) for CoreML: symbolic CRAFT + concrete enumerated shapes."""
    dims, example = _det_dyn()
    ep = te(DetNorm(load_craft(dynamic=True)), (torch.randn(1, 3, *example),),
            dynamic_shapes=({**dims},), strict=False)
    return ep, [[1, 3, h, w] for (h, w) in CRAFT_SHAPES]


def build_recognize_fixed(lang, w, *, int8):
    """ONE fixed-width unrolled `recognize` ep (nn.LSTM at concrete T fully delegates)."""
    if int8:
        return xnnpack.pt2e_int8(
            lambda: UnifiedCRNN(load_crnn(lang)),
            (torch.randn(1, 3, CRNN_H, w),), crnn_calib([w])[w], is_per_channel=True)
    return te(UnifiedCRNN(load_crnn(lang)), (torch.randn(1, 3, CRNN_H, w),), strict=False)


def _rec_enum(ep, w):
    """enum_map entry declaring the baked recognize width for the runtime snap."""
    return {ep.graph_signature.user_inputs[0]: [[1, 3, CRNN_H, w]]}


def emit_charset(lang, out_path):
    """Write `EasyOCR_<lang>.charset.txt` (JSON list) beside the PTE. charset[i] -> logit
    i+1, logit 0 = CTC blank. Sourced from the installed `easyocr` package; best-effort."""
    try:
        import easyocr.config as EC
    except Exception:
        print("    [note] easyocr package absent -> skipping charset emission", flush=True)
        return None
    charset = list(EC.recognition_models["gen2"][weights.EC_KEY[lang]]["characters"])
    p = os.path.join(os.path.dirname(out_path), f"EasyOCR_{lang}.charset.txt")
    with open(p, "w") as f:
        json.dump(charset, f, ensure_ascii=False)
    print(f"    wrote {os.path.basename(p)} ({len(charset)} chars, V={len(charset)+1})", flush=True)
    return p


def build_and_lower(backend, lang, out_path):
    """Build the fused dynamic PTE for `backend` + `lang` and write it to absolute
    `out_path`. Emits `EasyOCR_<lang>.charset.txt` alongside. Returns the .pte path."""
    assert os.path.isabs(out_path), "out_path must be absolute"
    if lang not in weights.CRNN:
        raise ValueError(f"unknown lang '{lang}' (have {list(weights.CRNN)})")

    if backend == "xnnpack":
        # detect narrow-band int8 dynamic + recognize static int8 per-channel @512.
        rec = build_recognize_fixed(lang, CRNN_FIXED_W, int8=True)
        eps = {"detect": build_detect_dynamic("int8"), "recognize": rec,
               **det_companion(h_bounds=(DET_XNN_MIN, DET_XNN_MAX),
                               w_bounds=(DET_XNN_MIN, DET_XNN_MAX))}
        path = lower_save(eps, [xnnpack.partitioner()], out_path,
                          transform_passes=[xnnpack.InsertContig()],
                          enum_map={"recognize": _rec_enum(rec, CRNN_FIXED_W)}, width=24)

    elif backend == "coreml":
        # ENUMERATED detect (weight-only int8, fp16 ANE, full-graph) + static
        # recognize @512; enum_shapes constants drive the runtime snap for both.
        det_ep, det_shapes = build_detect_enumerated()
        rec_ep = build_recognize_fixed(lang, CRNN_FIXED_W, int8=False)
        eps = {"detect": det_ep, "recognize": rec_ep}
        enum_map = {"detect": {det_ep.graph_signature.user_inputs[0]: det_shapes},
                    "recognize": _rec_enum(rec_ep, CRNN_FIXED_W)}
        parts = {"detect": [coreml.partitioner(
                     "fp16", quantize="int8", lower_full_graph=True,
                     extra_specs=[coreml.enumerated_spec(det_ep, enum_map["detect"])])],
                 "recognize": [coreml.partitioner("fp16", quantize="int8",
                                                  lower_full_graph=False)]}
        path = lower_save(eps, parts, out_path, enum_map=enum_map, width=24)

    elif backend == "vulkan":
        # MIXED-delegate: detect Vulkan fp16 dynamic via resize (int8 q8ta cannot
        # resize) + recognize int8 on XNNPACK @512 (the BiLSTM won't delegate to
        # Vulkan; unrolled at concrete T -> XNNPACK matmuls). Delegate CRAFT's FPN
        # upsamples to Vulkan (scale_factor path only, safe on Xclipse) — else
        # portable upsample splits detect into ~4 blobs of GPU<->CPU round-trips.
        vulkan.patch_upsample_resize()
        rec = build_recognize_fixed(lang, CRNN_FIXED_W, int8=True)
        eps = {"detect": build_detect_dynamic("fp32"), "recognize": rec,
               **det_companion()}
        parts = {"detect": [vulkan.partitioner(True)]}
        parts.update({n: [xnnpack.partitioner()] for n in eps if n != "detect"})
        path = lower_save(eps, parts, out_path,
                          transform_passes=[xnnpack.InsertContig()],
                          enum_map={"recognize": _rec_enum(rec, CRNN_FIXED_W)}, width=24)

    else:
        raise ValueError(f"unknown backend '{backend}'")

    emit_charset(lang, out_path)
    return path
