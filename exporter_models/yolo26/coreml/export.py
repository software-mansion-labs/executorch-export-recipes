from pathlib import Path

import torch
from torch.export import export as torch_export


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


# NOTE: fp16 only, deliberately, and never the Core ML GPU.
#
# This graph aborts the process on the Core ML GPU path: the MPSGraph MLIR
# compiler fails an internal assertion ("Error: MLIR pass manager failed",
# MPSGraphExecutable.mm:5070). It is an assertion, so SIGABRT, which no caller
# can catch -- the host process dies and on device the app dies with it.
#
# CORRECTION 2026-08-20: this was originally recorded as an fp32 problem, on the
# reasoning that a FLOAT32 compute_precision build forces the GPU path. Precision
# is not the cause. An fp16 build with compute_unit pinned to CPU_AND_GPU hits the
# identical assertion, reproduced at all three resolutions (384/512/640). The
# trigger is the GPU path itself; fp32 merely happened to route there. Anything
# that puts this graph on the Core ML GPU will abort, at any precision.
#
# The graph itself is fine. The identical export runs correctly under ALL,
# CPU_ONLY and CPU_AND_NE, and matches the torch reference to 1.5e-4 px on boxes
# and 1e-6 on scores. efficientnet-v2-s lowered with CPU_AND_GPU runs normally, so
# this is specific to this graph rather than a general Core ML GPU fault: an Apple
# MPSGraph compiler bug, not a bad export.
#
# fp16 is also the only precision the ANE accepts, so nothing useful is lost.
# The yolo26 <size>/coreml/config.json files intentionally list fp16 variants only.
def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from ultralytics import YOLO

    size = config.get("size") or "n"
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]

    yolo = YOLO(f"yolo26{size}.pt")
    model = _YoloDetection(yolo.model).eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    ep = torch_export(model, inputs)
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )
    # lower_full_graph=False so any op CoreML cannot take stays on the CPU
    # backend instead of failing the whole export.
    program = to_edge_transform_and_lower(
        ep,
        partitioner=[CoreMLPartitioner(compile_specs=specs, lower_full_graph=False)],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
