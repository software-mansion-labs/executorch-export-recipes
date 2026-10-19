import importlib.util
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch.export import export as torch_export

from exporter.backends import lower

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# int8 calibration: 300 COCO val2017 images, disjoint from the first 500 (by id)
# the int8 variants were evaluated on.
_COCO_DIR = Path("datasets/coco")
_COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
_EVAL_HOLDOUT = 500
_CALIBRATION_SIZE = 300

# Kept fp32: the first conv and the Segment head (inner.model[0] and [22]), except
# the head's mask prototype branch, and the wrapper's top-k/bmm post-processing.
_FP32_MODULES = ("inner.model.0", "inner.model.22")
_INT8_INSIDE = ("inner.model.22.proto",)


def _load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "fast_sam_xnnpack_wrapper", _MODEL_DIR / "_fastsam_wrapper.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fast_sam_xnnpack_wrapper"] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _calibrate(prepared, shape):
    import cv2

    images = _coco_val2017()
    names = sorted(p.name for p in images.glob("*.jpg"))[_EVAL_HOLDOUT:]
    for name in random.Random(0).sample(names, _CALIBRATION_SIZE):
        rgb = cv2.cvtColor(cv2.imread(str(images / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        # RNE FASTSAM_OPTS: resizeMode 'stretch', linear
        rgb = cv2.resize(rgb, (shape[-1], shape[-2]), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(rgb).permute(2, 0, 1)
        prepared(x.unsqueeze(0).float().div(255.0).contiguous())


def _under(node, module) -> bool:
    paths = [p for p, _ in node.meta.get("nn_module_stack", {}).values()]
    return any(p == module or p.startswith(module + ".") for p in paths)


def _kept_fp32(node) -> bool:
    # Wrapper post-processing ops stay fp32. Weights and inputs carry no module
    # stack and follow the op that consumes them.
    if node.op == "call_function" and not _under(node, "inner.model"):
        return True
    return any(_under(node, m) for m in _FP32_MODULES) and not any(
        _under(node, m) for m in _INT8_INSIDE
    )


def _strip(gm, drop):
    """Remove quantization annotations from `drop` nodes and from every node that
    shares qparams with one of them (propagated through share-ops like squeeze)."""
    from torchao.quantization.pt2e.quantizer import SharedQuantizationSpec

    removed = set()
    for n in gm.graph.nodes:
        if drop(n) and n.meta.pop("quantization_annotation", None) is not None:
            removed.add(n)
    changed = True
    while changed:
        changed = False
        for n in gm.graph.nodes:
            a = n.meta.get("quantization_annotation")
            if a is None:
                continue
            refs = []
            for qs in [*a.input_qspec_map.values(), a.output_qspec]:
                if isinstance(qs, SharedQuantizationSpec):
                    e = qs.edge_or_node
                    refs += list(e) if isinstance(e, tuple) else [e]
            if any(r in removed for r in refs):
                n.meta.pop("quantization_annotation")
                removed.add(n)
                changed = True


def _quantize(model, inputs, shape):
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    class _Quantizer(XNNPACKQuantizer):
        # The head's decode cats pixel boxes with sigmoid scores, and a shared int8
        # scale zeroes the scores. Annotations are stripped after the fact because
        # _annotate_cat ignores filter_fn (and so set_module_type(..., None)).
        def annotate(self, gm):
            gm = super().annotate(gm)
            for m in _FP32_MODULES + _INT8_INSIDE:
                assert any(_under(n, m) for n in gm.graph.nodes), f"no nodes under {m}"
            _strip(gm, _kept_fp32)
            return gm

    quantizer = _Quantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    quantizer.set_filter_function(lambda n: not _kept_fp32(n))
    prepared = prepare_pt2e(torch_export(model, inputs).module(), quantizer)
    with torch.no_grad():
        _calibrate(prepared, shape)
    return convert_pt2e(prepared)


def export(config: dict, out_dir: str) -> str:
    from ultralytics import FastSAM

    wrapper_mod = _load_wrapper()
    FastSAMInstanceSegmentation = wrapper_mod.FastSAMInstanceSegmentation

    size = config.get("size") or "s"
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])  # [1,3,640,640]
    method_size = shape[-1]

    inner = FastSAM(f"FastSAM-{size}.pt").model
    if variant["quantized"]:
        # The checkpoint ships unfused Conv+BN, and PT2E-quantized convs followed by
        # a BN do not lower to int8 weights (the .pte stays fp32-sized).
        inner = inner.fuse(verbose=False)
    model = FastSAMInstanceSegmentation(inner, method_size=method_size)
    model.eval()

    inputs = (torch.randn(*shape, dtype=torch.float32),)
    with torch.no_grad():
        model(*inputs)  # warmup: FastSAM head needs a forward to fix export-mode shapes

    if variant["quantized"]:
        model = _quantize(model, inputs, shape)

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
