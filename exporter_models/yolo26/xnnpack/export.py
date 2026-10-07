import random
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch.export import export as torch_export

from exporter.backends import lower

# int8 calibration: 300 COCO val2017 images, disjoint from the first 500 (by id)
# the int8 variants were evaluated on.
_COCO_DIR = Path("datasets/coco")
_COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
_EVAL_HOLDOUT = 500
_CALIBRATION_SIZE = 300

# Kept fp32: the first conv and the Detect head (yolo.model.model[0] and [-1]).
_FP32_MODULES = ("inner.model.0", "inner.model.23")


class _YoloDetection(torch.nn.Module):
    """Repo YOLOObjectDetection contract: packed [max_det,6] head -> (boxes, scores, labels)."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.inner.eval()
        self.inner.model[-1].export = True

    def forward(self, x):
        preds = self.inner(x)
        p = preds[0]
        return p[:, :4], p[:, 4], p[:, 5]


def _coco_val2017() -> Path:
    images = _COCO_DIR / "val2017"
    if not images.is_dir():
        _COCO_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = _COCO_DIR / "val2017.zip"
        urllib.request.urlretrieve(_COCO_VAL_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(_COCO_DIR)
        zip_path.unlink()
    return images


def _letterbox(rgb, height, width):
    """RNE resizeMode 'letterbox': centered, std::round size, pad 0, INTER_LINEAR."""
    import cv2
    import numpy as np

    h, w = rgb.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(np.floor(w * scale + 0.5)), int(np.floor(h * scale + 0.5))
    ox, oy = (width - nw) // 2, (height - nh) // 2
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[oy : oy + nh, ox : ox + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def _calibrate(prepared, shape):
    import cv2

    images = _coco_val2017()
    names = sorted(p.name for p in images.glob("*.jpg"))[_EVAL_HOLDOUT:]
    for name in random.Random(0).sample(names, _CALIBRATION_SIZE):
        rgb = cv2.cvtColor(cv2.imread(str(images / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(_letterbox(rgb, shape[-2], shape[-1])).permute(2, 0, 1)
        prepared(x.unsqueeze(0).float().div(255.0).contiguous())


def _under(node, module) -> bool:
    paths = [p for p, _ in node.meta.get("nn_module_stack", {}).values()]
    return any(p == module or p.startswith(module + ".") for p in paths)


def _kept_fp32(node) -> bool:
    return any(_under(node, m) for m in _FP32_MODULES)


def _quantize(model, inputs, shape):
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    class _Quantizer(XNNPACKQuantizer):
        # Detect must stay fp32: its decode cats pixel boxes with sigmoid scores and
        # a shared int8 scale zeroes the scores. The first conv in fp32 recovers most
        # of the remaining COCO mAP. Annotations are stripped after the fact because
        # _annotate_cat ignores filter_fn (and so set_module_type(..., None)).
        def annotate(self, gm):
            gm = super().annotate(gm)
            for m in _FP32_MODULES:
                assert any(_under(n, m) for n in gm.graph.nodes), f"no nodes under {m}"
            for n in filter(_kept_fp32, gm.graph.nodes):
                n.meta.pop("quantization_annotation", None)
            return gm

    quantizer = _Quantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    quantizer.set_filter_function(lambda n: not _kept_fp32(n))
    prepared = prepare_pt2e(torch_export(model, inputs).module(), quantizer)
    with torch.no_grad():
        _calibrate(prepared, shape)
    return convert_pt2e(prepared)


def export(config: dict, out_dir: str) -> str:
    from ultralytics import YOLO

    size = config.get("size") or "n"
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    yolo = YOLO(f"yolo26{size}.pt")
    model = _YoloDetection(yolo.model).eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    if variant["quantized"]:
        model = _quantize(model, inputs, shape)

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
