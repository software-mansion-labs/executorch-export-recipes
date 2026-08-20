# Core ML export, RE-ENABLED 2026-08-20 after the original disabling was found to
# rest on an invalid measurement.
#
# The note this replaces claimed, measured "on a real image" 2026-08-19:
#     coreml fp16    7.017 px landmark error
#     coreml fp32   46.796 px + selects a different top-scoring query
# and concluded Core ML was unusable for this model.
#
# That measurement was wrong. The test fed a NON-CONTIGUOUS input tensor
# (PIL -> numpy -> .permute(2,0,1) with no .contiguous()). ExecuTorch reads the
# underlying storage in the wrong order and returns deterministic garbage, while
# eager PyTorch handles the same tensor correctly -- so the reference looks right
# and the .pte looks broken. The same mistake was independently repeated on
# 2026-08-20 and produced the same false negative, which is how it was caught.
#
# Re-measured with .contiguous() input, published fp32 artifact, 6 real VOC person
# images, against the float reference:
#     box_err 0.000   score_err 0.0000   landmark err <= 0.046 px   6/6 images
# and confirmed WORKING on a physical iPhone in the demo app.
#
# ALWAYS call .contiguous() on image tensors before Method.execute(). This class of
# bug silently condemns any image model.
#
# Caveat carried forward, NOT re-verified: the fp16 variant was reported at
# 7.017 px and looked visibly wrong on device. That figure came from the same
# invalid harness, so treat it as unmeasured rather than as a known defect. Only
# fp32 is published and verified; re-test fp16 with contiguous input before
# shipping it.

import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_model():
    name = "rfdetr_kp_model_coreml"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_keypoint_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        config["variant"]["precision"]
    ]
    wrapper, example = _load_model().build_wrapper()
    with torch.inference_mode():
        ep = torch_export(wrapper, args=example, strict=False)

    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18, compute_precision=precision
    )
    program = to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
