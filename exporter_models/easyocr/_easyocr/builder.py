"""EasyOCR (CRAFT detector + per-language CRNN recognizer) — fused DYNAMIC exporter.

Builds ONE fused .pte per backend + language: `detect` (CRAFT) + fixed-width
`recognize` (CRNN), plus the model-wide `get_model_schema` constant method — a
no-arg method returning ONE JSON ModelSpec string (see model.schema.json). `detect`
carries the dynamic/enumerated H,W domains; the all-static `recognize` is spelled
out only to declare its width/timestep runtime constraint (W = 4*T + 4).
The CRNN's BiLSTM cannot be dynamic on CPU/ANE (symbolic T -> torch.scan -> portable
~15s; concrete T -> nn.LSTM unrolls into matmuls that fully delegate), so the C++
snaps every crop to the baked width.

  CRAFT (detect)  [1,3,H,W] f32, ImageNet-NORMALIZED RGB (client computes
                  (x/255 - mean)/std; NOTHING baked)
                  -> [1,H/2,W/2,2] score NHWC (heatmap ONLY; RefineNet feat dropped)
                  H,W dynamic, step 32 — per-backend band (see get_model_schema):
                  xnnpack [320, 1280] both dims (accurate <=800, best-effort above);
                  vulkan H [800, 1280], W [320, 1280]; coreml enumerated sparse set
  CRNN  (recognize) [1,3,64,512] f32 RGB, client norm (x/255 - 0.5)/0.5 as always
                  -> [1, 127, V] probs, ONE fixed width
                  (RGB->gray 1x1 conv + softmax BAKED -> same contract as SVTR)

Per-backend precision + delegate (fixed; no knob):
  xnnpack : detect static-act int8 dynamic [320,1280], calib <=800 (qs8 chains =
            bucketed-class speed) + recognize int8 @512.
  coreml  : ENUMERATED GRID detect (weight-only int8, fp16 ANE) + recognize @512
            static (no RangeDim on CoreML; per-dim enum == grid).
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

# Dynamic bounds for the fp16 (Vulkan) detect. The H floor used to be 800, not
# because Vulkan needs it — the device-level cap is the 3D-texture extent, held
# separately by texture_limits, and fp16 resizes losslessly — but because the
# legacy bucketed artifact only ever shipped 800 and 1280 buckets, so the range
# that replaced it started where the validated buckets did. That floor costs
# real work on short pages: the client snaps H up to it, so a 1280x400 page runs
# 1280x800, ~1.9x the pixels it needs, on the slowest detect we ship. Floored at
# 320 (= DET_W_MIN) so H and W cover the same band.
# NOTE: fp16 CRAFT has never been parity-checked below H=800; treat the new part
# of the band as unverified until it is.
DET_H_MIN, DET_H_MAX = 320, 1280
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
# CoreML enumerated det = the full DET_CML_SIZES x DET_CML_SIZES grid (per-dim enum
# in get_model_schema implies dim independence).
DET_CML_SIZES = [320, 800, 1280]
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


# ── recognizer bakes (detector norm is CLIENT-SIDE — nothing baked in `detect`) ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# `detect` expects an ALREADY-NORMALIZED float32 [1,3,H,W] RGB input: the client
# computes (x/255 - mean)/std per channel (standard ImageNet stats above). No norm
# wrapper or weight-fold is baked. (Historical note: an in-graph broadcast sub/div
# norm in front of a quantize convert trips an ET XNNPACK delegate bug —
# xnn_reshape_runtime xnn_status_invalid_parameter — which is why the previous
# baked-norm contract needed a conv1 weight-fold; client-side norm avoids it all.)


class DetectY(nn.Module):
    """CRAFT's forward returns (y, feature); the [1,32,H/2,W/2] RefineNet feature
    map is consumed by nothing on-device — export ONLY the detection heatmap
    y [1,H/2,W/2,2] (region + affinity, NHWC)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)[0]


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
    """Real-page calibration in the detector's input domain: ImageNet-normalized
    RGB — the client applies (x/255 - mean)/std, nothing is baked in the graph."""
    import cv2
    import numpy as np
    mean = np.array(IMAGENET_MEAN, np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, np.float32).reshape(1, 1, 3)
    out = {}
    for (h, w) in shapes:
        xs = []
        for i in range(1, 5):
            p = os.path.join(HERE, "calibration_images", f"{i}.png")
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            r = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            r = (r - mean) / std
            xs.append(torch.from_numpy(r.transpose(2, 0, 1)[None].copy()))
        out[(h, w)] = xs or [torch.randn(1, 3, h, w) for _ in range(8)]
    return out


# ── dynamic dims + schema ──
def _det_dyn():
    """(dynamic_shapes dict, example (H,W)) for the symbolic CRAFT export."""
    h = Dim("det_h", min=DET_H_MIN // CRAFT_DIV, max=DET_H_MAX // CRAFT_DIV)
    w = Dim("det_w", min=DET_W_MIN // CRAFT_DIV, max=DET_W_MAX // CRAFT_DIV)
    return {2: CRAFT_DIV * h, 3: CRAFT_DIV * w}, (DET_H_MAX, DET_W_MAX)


def _C(v): return {"kind": "constant", "value": v}
def _R(lo, hi, step): return {"kind": "range", "range": {"min": lo, "max": hi, "step": step}}
def _E(choices): return {"kind": "enum", "choices": list(choices)}
def _T(dims): return {"kind": "Tensor", "dtype": "float32", "shape": dims}
def _in(t, d): return {"paramSide": "input", "tensorIdx": t, "dimIdx": d}
def _out(t, d): return {"paramSide": "output", "tensorIdx": t, "dimIdx": d}
def _lin(lhs, rhs, a, b):
    """Runtime constraint `lhs = a * rhs + b` over two tensor dims."""
    return {"kind": "linear", "dimLhs": lhs, "dimRhs": rhs, "coefficients": [a, b]}


def _validate(spec):
    """Inline validation against the repo root model.schema.json (same check as
    exporter.schema.load_and_validate_schema, for programmatic specs)."""
    import jsonschema
    from exporter.schema import _ROOT_SCHEMA_PATH
    jsonschema.validate(spec, json.loads(_ROOT_SCHEMA_PATH.read_text()))


def model_schema(backend, lang):
    """JSON ModelSpec baked as the `get_model_schema` constant method.

    `detect` carries the per-backend dynamic/enumerated H,W domains. `recognize` is
    all-static ([1,3,64,512] -> [1,127,V]) and would fall back to MethodMeta, but it
    is spelled out anyway to carry its ONE runtime constraint: the CRNN crops a
    trailing timestep, so W = 4*T + 4 (NOT a plain W/T ratio like SVTR's). Declared
    so the client can pre-size the probs output from the width it picks instead of
    inferring a ratio that does not exist here.
    Per-dim enum implies dim independence -> the CoreML det set is the full grid."""
    if backend == "coreml":
        hw = _E(DET_CML_SIZES)
        out_hw = _E(sorted(v // 2 for v in DET_CML_SIZES))
        h, w, oh, ow = hw, hw, out_hw, out_hw
    else:
        h_lo = DET_XNN_MIN if backend == "xnnpack" else DET_H_MIN
        h = _R(h_lo, 1280, CRAFT_DIV)
        w = _R(DET_W_MIN if backend == "vulkan" else DET_XNN_MIN, 1280, CRAFT_DIV)
        oh = _R(h["range"]["min"] // 2, 640, CRAFT_DIV // 2)
        ow = _R(w["range"]["min"] // 2, 640, CRAFT_DIV // 2)
    timesteps = CRNN_FIXED_W // CRNN_DIV - 1
    spec = {
        "detect": {"inputs": [_T([_C(1), _C(3), h, w])],
                   "outputs": [_T([_C(1), oh, ow, _C(2)])],
                   "runtimeConstraints": []},
        "recognize": {"inputs": [_T([_C(1), _C(3), _C(CRNN_H), _C(CRNN_FIXED_W)])],
                      "outputs": [_T([_C(1), _C(timesteps), _C(weights.num_class(lang))])],
                      "runtimeConstraints": [_lin(_in(0, 3), _out(0, 1), CRNN_DIV, CRNN_DIV)]},
    }
    _validate(spec)
    return json.dumps(spec)


# ── per-stage export-program builders ──
def build_detect_dynamic(precision):
    """ONE symbolic-dim `detect` ep over RAW CRAFT (input is client-normalized).

    int8 (XNNPACK) = static-act int8 over [320, 1280], calibrated on real pages
    (ImageNet-normalized domain) at DET_XNN_BAND sizes — accurate <=800, best-effort
    above. qs8 int8-int8 chains run at bucketed-class speed — vs qd8 (1.6x slower)
    or fp32 (5x, and fp16 is fake on XNNPACK: fp32-sized AND fp32-speed).

    fp32 (Vulkan) = plain graph over the full [800..1280 x 320..1280] range
    (fp16 via force_fp16 resizes losslessly; never .half())."""
    if precision == "int8":
        h = Dim("det_h", min=DET_XNN_MIN // CRAFT_DIV, max=DET_XNN_MAX // CRAFT_DIV)
        w = Dim("det_w", min=DET_XNN_MIN // CRAFT_DIV, max=DET_XNN_MAX // CRAFT_DIV)
        dims = {2: CRAFT_DIV * h, 3: CRAFT_DIV * w}
        cmap = craft_calib(DET_XNN_BAND)
        cal = [x for s in DET_XNN_BAND for x in cmap[s][:4]]
        return xnnpack.pt2e_int8(
            DetectY(load_craft(dynamic=True)).eval(),
            (torch.randn(1, 3, DET_XNN_MAX, DET_XNN_MAX),), cal,
            is_per_channel=True, dynamic_shapes=dims)
    dims, example = _det_dyn()
    return te(DetectY(load_craft(dynamic=True)), (torch.randn(1, 3, *example),),
              dynamic_shapes=({**dims},), strict=False)


def build_detect_enumerated():
    """(ep, enum shape list) for CoreML: symbolic RAW CRAFT over [320,1280]² + the
    full grid of concrete shapes."""
    lo, hi = DET_CML_SIZES[0], DET_CML_SIZES[-1]
    h = Dim("det_h", min=lo // CRAFT_DIV, max=hi // CRAFT_DIV)
    w = Dim("det_w", min=lo // CRAFT_DIV, max=hi // CRAFT_DIV)
    dims = {2: CRAFT_DIV * h, 3: CRAFT_DIV * w}
    ep = te(DetectY(load_craft(dynamic=True)), (torch.randn(1, 3, hi, hi),),
            dynamic_shapes=({**dims},), strict=False)
    return ep, [[1, 3, h_, w_] for h_ in DET_CML_SIZES for w_ in DET_CML_SIZES]


def build_recognize_fixed(lang, w, *, int8):
    """ONE fixed-width unrolled `recognize` ep (nn.LSTM at concrete T fully delegates)."""
    if int8:
        return xnnpack.pt2e_int8(
            lambda: UnifiedCRNN(load_crnn(lang)),
            (torch.randn(1, 3, CRNN_H, w),), crnn_calib([w])[w], is_per_channel=True)
    return te(UnifiedCRNN(load_crnn(lang)), (torch.randn(1, 3, CRNN_H, w),), strict=False)


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
        eps = {"detect": build_detect_dynamic("int8"), "recognize": rec}
        path = lower_save(eps, [xnnpack.partitioner()], out_path,
                          transform_passes=[xnnpack.InsertContig()],
                          extra_constants={"get_model_schema": model_schema(backend, lang)},
                          width=24)

    elif backend == "coreml":
        # ENUMERATED-GRID detect (weight-only int8, fp16 ANE, full-graph) + static
        # recognize @512; discovery via the get_model_schema JSON constant.
        det_ep, det_shapes = build_detect_enumerated()
        rec_ep = build_recognize_fixed(lang, CRNN_FIXED_W, int8=False)
        eps = {"detect": det_ep, "recognize": rec_ep}
        spec_map = {det_ep.graph_signature.user_inputs[0]: det_shapes}
        parts = {"detect": [coreml.partitioner(
                     "fp16", quantize="int8", lower_full_graph=True,
                     extra_specs=[coreml.enumerated_spec(det_ep, spec_map)])],
                 "recognize": [coreml.partitioner("fp16", quantize="int8",
                                                  lower_full_graph=False)]}
        path = lower_save(eps, parts, out_path,
                          extra_constants={"get_model_schema": model_schema(backend, lang)},
                          width=24)

    elif backend == "vulkan":
        # MIXED-delegate: detect Vulkan fp16 dynamic via resize (int8 q8ta cannot
        # resize) + recognize int8 on XNNPACK @512 (the BiLSTM won't delegate to
        # Vulkan; unrolled at concrete T -> XNNPACK matmuls). Delegate CRAFT's FPN
        # upsamples to Vulkan (scale_factor path only, safe on Xclipse) — else
        # portable upsample splits detect into ~4 blobs of GPU<->CPU round-trips.
        vulkan.patch_upsample_resize()
        rec = build_recognize_fixed(lang, CRNN_FIXED_W, int8=True)
        eps = {"detect": build_detect_dynamic("fp32"), "recognize": rec}
        parts = {"detect": [vulkan.partitioner(True)],
                 "recognize": [xnnpack.partitioner()]}
        path = lower_save(eps, parts, out_path,
                          transform_passes=[xnnpack.InsertContig()],
                          extra_constants={"get_model_schema": model_schema(backend, lang)},
                          width=24)

    else:
        raise ValueError(f"unknown backend '{backend}'")

    emit_charset(lang, out_path)
    return path
