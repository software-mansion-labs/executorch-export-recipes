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


# DEVICE WARNING (RESOLVED 2026-08-28): fp16 is broken for this model on iPhone
# under ALL, CPU_ONLY and CPU_AND_NE, and the failure is invisible to host
# testing. It is NOT broken under CPU_AND_GPU, which export() now selects for
# fp16; see the compute-unit note there. The original text follows, and its
# conclusion that "it is not the Neural Engine" was half right -- on device both
# the CPU and the ANE fp16 paths are wrong, so the fix is to exclude the CPU
# *and* the ANE rather than to give up on fp16.
#
# Original note:
#
# Measured 2026-08-20 on an iPhone 16: the fp32 build detects and draws keypoints
# correctly; the fp16 build draws nothing at all. On macOS both builds are
# indistinguishable (max score 0.813 vs 0.811 and 0.868 vs 0.867 on the same
# person images), so a host-side check passes fp16 cleanly.
#
# It is NOT the Apple Neural Engine. The obvious hypothesis was that fp16 reaches
# the ANE where fp32 cannot, so an fp16 build pinned to compute_unit=CPU_ONLY was
# also tested on device: it fails identically. Whatever breaks is in the fp16
# numerics or lowering of this graph, not in compute-unit routing.
#
# Consequences:
#   * Do not ship an fp16 Core ML variant for this model. fp32 stays.
#   * A latency benchmark cannot be trusted here on its own. fp16 measured 1.30x
#     faster than fp32 (123.4 vs 159.9 ms) at half the size while producing
#     unusable output, because the benchmark harness verifies that output is
#     finite and correctly delegated, not that it is correct.


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

    # DO NOT set iOS18 here. It silently degrades this model.
    #
    # Swept on host against the eager reference (max person score, person2.jpg):
    #     target unset  0.8650      iOS16  0.8650
    #     iOS17         0.8650      iOS18  0.3056
    # Boxes are unaffected at every target -- they are anchored by the decoder's
    # reference points -- so only scores and landmarks move, which is why a
    # visual check of the boxes passes and the model still looks plausible.
    #
    # This reproduces under the macOS ExecuTorch runtime, so it needs no device:
    #     from executorch.runtime import Runtime
    #     method = Runtime.get().load_program(pte).load_method("forward")
    #     _, scores, _ = method.execute([x])          # x must be .contiguous()
    # The published fp32 artifact scores 0.8650 through that same path and is
    # ~150 KB larger than an iOS18 build, which is the size tell. It predates
    # 171c4fd (2026-07-30), where iOS18 was added across 31 export scripts.
    #
    # iOS17 is a floor, not a ceiling: the artifact still runs on iOS 18+.
    #
    # This does NOT rehabilitate fp16. Retested at iOS17 on an iPhone 16: fp32
    # scores 0.863, matching the published artifact exactly, while fp16 scores
    # 0.422 with landmarks 50.8 px out. On host the same fp16 build looks fine
    # (0.8613, within 0.32 px of the eager reference), so the DEVICE WARNING
    # above still stands and the iOS18 bug was a second, independent defect.
    # fp16 is correct on device ONLY with the CPU excluded. Measured at iOS17 on
    # an iPhone 16 against the fp32 build's 0.863:
    #     all 0.421   cpu_only 0.411   cpu_and_ne 0.422   cpu_and_gpu 0.862
    # so both the CPU and the Neural Engine fp16 paths return wrong scores for
    # this graph and the GPU is the only one that does not. On macOS only the
    # CPU path is wrong, which is why a host check passes fp16 under ALL and the
    # device still fails -- the two platforms place ALL differently.
    #
    # With cpu_and_gpu, fp16 matches fp32 to 0.0008 confidence, 0.23 px box and
    # 0.17 px landmark, at 144.0 ms against 165.1 ms and 263 MB against 389 MB.
    compute_unit = ct.ComputeUnit.CPU_AND_GPU if precision == ct.precision.FLOAT16 else ct.ComputeUnit.ALL

    specs = CoreMLBackend.generate_compile_specs(
        compute_unit=compute_unit,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
    )
    program = to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
