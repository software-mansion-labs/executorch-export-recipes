"""EasyOCR (CRAFT detector + per-language CRNN recognizer) — fused bucketed exporter.

Builds ONE fused .pte (detect + recognize) of static per-size methods plus an
`is_bucketed()` constant, for a given backend + language. Ported from the standalone
`export_easyocr.py` (the `--bucketed` fused path); the argparse / config-emission
shell is gone — the harness authors config.json and calls
`build_and_lower(backend, lang, out_path)`.

  CRAFT (detect)  [1,3,H,W] f32 [0,1] RGB -> ([1,H/2,W/2,2] score, [1,32,H/2,W/2] feat)
                  (ImageNet norm BAKED -> runtime feeds raw [0,1] RGB; unifies w/ DBNet)
  CRNN  (recognize) [1,3,64,W] f32 RGB    -> [1, W/4-1, V] probs
                  (RGB->gray 1x1 conv + softmax BAKED -> same contract as SVTR)

Per-backend precision + delegate (fixed; no knob):
  xnnpack : CRAFT int8 (per-tensor) + CRNN int8 (per-channel), all on XNNPACK CPU.
  coreml  : weight-only int8, fp16 compute (ANE); static full-graph per method.
  vulkan  : MIXED — CRAFT int8 per-channel on Vulkan GPU (q8ta: per-channel weights +
            force_fp16=False), CRNN int8 on XNNPACK CPU (the BiLSTM won't delegate to
            Vulkan; bucketed -> unrolled at concrete T -> XNNPACK matmuls).

CRNN is exported UNROLLED per bucket (nn.LSTM specializes the time dim and fully
delegates at concrete T). The dynamic-time-dim `scan` BiLSTM (arch.ScanBiLSTM) is
kept in the vendored arch for the alternative true-dynamic deliverable, but the
shipped bucketed fallback does not use it.
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

CRAFT_SHAPES = [(800, 800), (1280, 1280), (1280, 320)]   # (H, W) detector buckets
CRNN_WIDTHS = [64, 128, 256, 512]
CRNN_H = 64
CRNN_FIXED_W = 512   # CoreML fixed width (ANE @512 fast)  — enumerated-path leftover
CRNN_XNN_W = 256     # XNNPACK fixed width                 — enumerated-path leftover
CRAFT_DIV, CRNN_DIV = 32, 4                               # spatial divisors


# ── weight loaders ──
def load_craft(dynamic):
    m = (CRAFTDyn if dynamic else CRAFT)()
    m.load_state_dict(copyStateDict(torch.load(
        str(weights.craft_path()), map_location="cpu", weights_only=False)))
    return m.eval()


def load_crnn(lang, dynamic=False):
    m = CRNNModel(input_channel=1, output_channel=256, hidden_size=256,
                  num_class=weights.num_class(lang))
    sd = torch.load(str(weights.crnn_path(lang)), map_location="cpu", weights_only=False)
    m.load_state_dict(OrderedDict((k[7:], v) for k, v in sd.items())); m.eval()
    if dynamic:
        m.SequenceModeling = nn.Sequential(
            ScanBiL(m.SequenceModeling[0]), ScanBiL(m.SequenceModeling[1]))
    return m.eval()


class CrnnFp16(nn.Module):
    """CRNN in fp16 (testing variant)."""
    def __init__(self, m): super().__init__(); self.m = m.half()
    def forward(self, x): return self.m(x.half()).float()


def _wrap(model, precision, stage):
    if precision == "fp16" and stage == "crnn":
        return CrnnFp16(model)
    return model


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


def export_xnnpack_int8(load_fn, stage, items, dynamic, calib):
    """PT2E static int8 (CRAFT per-tensor, CRNN per-channel); orchestrates the bucket loop."""
    per_channel = (stage == "crnn")
    ch = 3   # both: CRAFT RGB, CRNN now RGB (gray-conv baked)
    eps = {}
    def cal_for(h, w): return calib.get((h, w)) or calib.get(w) or next(iter(calib.values()))
    if dynamic:
        h, w = items["example"]; dims = items["dims"]
        eps[items["mname"]] = xnnpack.pt2e_int8(
            load_fn, (torch.randn(1, ch, h, w),), cal_for(h, w),
            is_per_channel=per_channel, dynamic_shapes=dims)
    else:
        for (h, w), mname in items:
            eps[mname] = xnnpack.pt2e_int8(
                load_fn, (torch.randn(1, ch, h, w),), cal_for(h, w),
                is_per_channel=per_channel)
    return eps


def _det_name(h, w):
    return f"detect_{w}" if h == w else f"detect_{w}x{h}"


# ── build one stage -> (eps, enum_map) ──
def build_stage(stage, backend, precision, mode, lang, crnn_w=None):
    if stage == "craft":
        shapes, div, ch = CRAFT_SHAPES, CRAFT_DIV, 3
        name_static = _det_name
        mname, axes = "detect", {2: True, 3: True}
        loader = lambda: DetNorm(load_craft(dynamic=(mode != "static")))   # bake norm -> [0,1] in
    else:
        shapes, div, ch = [(CRNN_H, w) for w in CRNN_WIDTHS], CRNN_DIV, 3   # 3ch: gray-conv baked
        name_static = lambda h, w: f"recognize_{w}"
        mname, axes = "recognize", {3: True}
        loader = lambda: UnifiedCRNN(load_crnn(lang, dynamic=(mode != "static")))  # 3ch RGB->probs
    model = loader()

    if mode == "static":
        if backend == "xnnpack" and precision == "int8":
            calib = craft_calib(shapes) if stage == "craft" else crnn_calib(CRNN_WIDTHS)
            return export_xnnpack_int8(loader, stage,
                [(s, name_static(*s)) for s in shapes], dynamic=False, calib=calib), None
        m = _wrap(model, precision, stage)
        return {name_static(h, w): te(m, (torch.randn(1, ch, h, w),), strict=False)
                for (h, w) in shapes}, None

    # CRNN enumerated = ONE fixed-width unrolled `recognize` for XNNPACK/CoreML.
    if backend in ("xnnpack", "coreml") and mode == "enumerated" and stage == "crnn":
        h = CRNN_H
        w = crnn_w or (CRNN_XNN_W if backend == "xnnpack" else CRNN_FIXED_W)
        if backend == "xnnpack" and precision == "int8":
            eps = export_xnnpack_int8(lambda: UnifiedCRNN(load_crnn(lang, dynamic=False)), stage,
                                      [((h, w), mname)], dynamic=False, calib=crnn_calib([w]))
        else:
            m = _wrap(UnifiedCRNN(load_crnn(lang, dynamic=False)), precision, stage)
            eps = {mname: te(m, (torch.randn(1, ch, h, w),), strict=False)}
        in_name = eps[mname].graph_signature.user_inputs[0]
        return eps, {mname: {in_name: [[1, ch, h, w]]}}

    # dynamic / enumerated
    def derived(name, vals):
        lo, hi = min(vals), max(vals)
        return div * Dim(name, min=lo // div, max=hi // div) if hi > lo else lo
    hs = sorted({h for h, _ in shapes}); ws = sorted({w for _, w in shapes})
    dims = {}
    if 2 in axes and len(hs) > 1: dims[2] = derived(f"{stage}_h", hs)
    dims[3] = derived(f"{stage}_w", ws)
    dims = {k: v for k, v in dims.items() if not isinstance(v, int)}
    example = (max(hs), max(ws))

    if backend in ("xnnpack", "vulkan"):
        if backend == "xnnpack" and precision == "int8":
            calib = craft_calib([example]) if stage == "craft" else crnn_calib([example[1]])
            return export_xnnpack_int8(loader, stage,
                {"dims": dims, "example": example, "mname": mname}, dynamic=True, calib=calib), None
        m = _wrap(model, precision, stage)
        return {mname: te(m, (torch.randn(1, ch, *example),), dynamic_shapes=({**dims},), strict=False)}, None

    # coreml -> enumerated
    m = _wrap(model, precision, stage)
    ep = te(m, (torch.randn(1, ch, *example),), dynamic_shapes=({**dims},), strict=False)
    in_name = ep.graph_signature.user_inputs[0]
    return {mname: ep}, {mname: {in_name: [[1, ch, h, w] for (h, w) in shapes]}}


def bucket_metadata(craft_shapes, crnn_widths):
    """is_bucketed() -> int32[2,N]: row0 CRAFT square sizes, row1 CRNN widths (-1 padded)."""
    det = sorted({w for (h, w) in craft_shapes if h == w})
    rows = [det, sorted(crnn_widths)]
    w = max(len(r) for r in rows)
    rows = [r + [-1] * (w - len(r)) for r in rows]
    return {"is_bucketed": torch.tensor(rows, dtype=torch.int32)}


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
    """Build the fused bucketed PTE for `backend` + `lang` and write it to absolute
    `out_path`. Emits `EasyOCR_<lang>.charset.txt` alongside. Returns the .pte path."""
    assert os.path.isabs(out_path), "out_path must be absolute"
    if lang not in weights.CRNN:
        raise ValueError(f"unknown lang '{lang}' (have {list(weights.CRNN)})")
    consts = bucket_metadata(CRAFT_SHAPES, CRNN_WIDTHS)

    if backend == "xnnpack":
        # CRAFT int8 (per-tensor) + CRNN int8 (per-channel), all XNNPACK CPU.
        craft_eps, _ = build_stage("craft", "xnnpack", "int8", "static", None)
        crnn_eps, _ = build_stage("crnn", "xnnpack", "int8", "static", lang)
        eps = {**craft_eps, **crnn_eps}
        path = lower_save(eps, [xnnpack.partitioner()], out_path,
                          transform_passes=[xnnpack.InsertContig()], extra_constants=consts)

    elif backend == "coreml":
        # weight-only int8, fp16 compute (ANE); static methods -> full-graph per method.
        craft_eps, _ = build_stage("craft", "coreml", "int8", "static", None)
        crnn_eps, _ = build_stage("crnn", "coreml", "int8", "static", lang)
        eps = {**craft_eps, **crnn_eps}
        parts = {n: [coreml.partitioner("fp16", quantize="int8", lower_full_graph=True)]
                 for n in eps}
        path = lower_save(eps, parts, out_path, extra_constants=consts)

    elif backend == "vulkan":
        # MIXED-delegate: CRAFT int8 per-channel on Vulkan (q8ta needs per-channel weights +
        # force_fp16=False — smaller AND faster than fp16 on a memory-bound net), CRNN int8 on
        # XNNPACK (the BiLSTM won't delegate to Vulkan; unrolled at concrete T -> XNNPACK matmuls).
        craft_eps = {
            _det_name(h, w): xnnpack.pt2e_int8(
                lambda h=h, w=w: DetNorm(load_craft(dynamic=False)),
                (torch.randn(1, 3, h, w),), craft_calib([(h, w)])[(h, w)],
                is_per_channel=True)
            for (h, w) in CRAFT_SHAPES}
        crnn_eps, _ = build_stage("crnn", "xnnpack", "int8", "static", lang)
        eps = {**craft_eps, **crnn_eps}
        parts = {n: [vulkan.partitioner(False, force_fp16=False)] for n in craft_eps}
        parts.update({n: [xnnpack.partitioner()] for n in crnn_eps})
        path = lower_save(eps, parts, out_path,
                          transform_passes=[xnnpack.InsertContig()], extra_constants=consts)

    else:
        raise ValueError(f"unknown backend '{backend}'")

    emit_charset(lang, out_path)
    return path
