# DISABLED: Core ML is not supported for this model, and this whole exporter is
# commented out. Both precisions produce wrong keypoint coordinates -- boxes come
# out approximately right, landmarks do not. Measured 2026-08-19 against the float
# PyTorch reference on a real image (ExecuTorch 1.4.1 / coremltools 9.0), max
# landmark error at the reference's best query:
#
#     xnnpack fp32    0.000 px   (exact)
#     coreml  fp16    7.017 px
#     coreml  fp32   46.796 px   + selects a different top-scoring query
#
# fp16 was wired into the demo app and still looked visibly broken on device, so
# Core ML is disabled outright rather than shipped at the better precision.
#
# fp32 being WORSE than fp16 points at a coremltools compilation bug, not rounding.
# Ruled out: the GPU path (CPU_ONLY and CPU_AND_NE reproduce it) and grid_sample
# (the gather-based grid-free deformable core reproduces it exactly). The offending
# op has NOT been identified; to resume, dump intermediate activations through the
# keypoint head and find the divergence point.
#
# Use MLX instead on Apple hardware: it is numerically correct and ~5x faster than
# XNNPACK on device. "coreml" is also removed from this model's model.json backends
# so batch exports skip it. The code below is preserved verbatim, commented, so the
# finding can be re-tested against a future coremltools / ExecuTorch: uncomment it
# and re-add "coreml" to model.json.

# import importlib.util
# import sys
# from pathlib import Path
#
# import torch
# from torch.export import export as torch_export
#
# _HERE = Path(__file__).resolve().parent
# _MODEL_DIR = _HERE.parent
#
#
# def _load_model():
#     name = "rfdetr_kp_model_coreml"
#     spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_keypoint_model.py")
#     mod = importlib.util.module_from_spec(spec)
#     sys.modules[name] = mod
#     spec.loader.exec_module(mod)
#     return mod
#
#
# def export(config: dict, out_dir: str) -> str:
#     import coremltools as ct
#     from executorch.backends.apple.coreml.compiler import CoreMLBackend
#     from executorch.backends.apple.coreml.partition import CoreMLPartitioner
#     from executorch.exir import to_edge_transform_and_lower
#
#     precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
#         config["variant"]["precision"]
#     ]
#     wrapper, example = _load_model().build_wrapper()
#     with torch.inference_mode():
#         ep = torch_export(wrapper, args=example, strict=False)
#
#     specs = CoreMLBackend.generate_compile_specs(
#         minimum_deployment_target=ct.target.iOS18, compute_precision=precision
#     )
#     program = to_edge_transform_and_lower(
#         ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
#     ).to_executorch()
#
#     out_path = Path(out_dir) / config["variant"]["file"]
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_bytes(program.buffer)
#     return str(out_path)
