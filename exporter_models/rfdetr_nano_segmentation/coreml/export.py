import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _apply_patches():
    spec = importlib.util.spec_from_file_location("rfdetr_seg_patches", _MODEL_DIR / "_patches.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_patches()


class _RFDETRSegmentationWrapper(nn.Module):
    def __init__(self, model: nn.Module, resolution: int):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def forward(self, x: torch.Tensor):
        pred_boxes, pred_logits, pred_masks = self.model(x)
        cxy = pred_boxes[..., :2]
        wh = pred_boxes[..., 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        bboxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution
        probs = pred_logits.sigmoid()
        max_scores, class_ids = probs.max(dim=-1)
        # flattened contract: 4 batch-squeezed outputs (boxes, scores, labels, masks),
        # matching the xnnpack export and createInstanceSegmenter's allowed spec
        # [f32(N,4), f32(N), f32(N), f32(N,MH,MW)]. This exporter previously packed
        # scores and class ids into one (1,N,2) tensor and kept the batch dim, which
        # no variant of the task contract accepts.
        #
        # squeeze(0), NOT `[0]`. A rank-reducing select on the delegate's outputs
        # silently corrupts Core ML's CPU path: geometry collapses (a 15x26 box
        # where the truth is 134x291) and 87 of 100 boxes come back exactly
        # [0, 0, 0, 0], while the classification head stays perfectly correct.
        # Measured on an iPhone 16, iOS 26.6.1, fp16, CPU_ONLY, 2026-09-04:
        #
        #   x[0]           IoU 0.0025   <- what this used to emit
        #   x.squeeze(0)   IoU 0.9957
        #   x.reshape(...)  IoU 0.9957
        #   batch dim kept IoU 0.9957
        #
        # Bisected to this one variable: splitting scores from class ids makes no
        # difference, the batch squeeze is the whole trigger. See
        # software-mansion/react-native-executorch#1406.
        num_queries = bboxes.shape[1]
        return (
            bboxes.squeeze(0),
            max_scores.squeeze(0),
            class_ids.float().squeeze(0),
            pred_masks.squeeze(0),
        )


# CONTRACT: 4 batch-squeezed outputs (boxes, scores, labels, masks), matching
# createInstanceSegmenter's allowed spec [f32(N,4), f32(N), f32(N), f32(N,MH,MW)]
# and the xnnpack export.
#
# Until 2026-08-20 this exporter emitted three batched outputs instead --
# [(1,100,4), (1,100,2), (1,100,78,78)] -- packing scores and class ids together
# and keeping the batch dim. No variant of the task contract accepts that, so the
# model failed spec validation on device with "spec doesn't match any of provided
# variant for batched and unbatched". It affected every precision, so the Core ML
# build of this model had never been usable through the task.

def _lower(exported, precision: str):
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    # fp16, not int8. The previous int8 build was weight-only quantization on top
    # of compute_precision=FLOAT32, which got the accuracy cost of quantization
    # and none of the benefit: fp32 compute is the one precision the ANE refuses,
    # so the model was locked onto CPU/GPU. Measured on an iPhone 16 (3 warmup +
    # 15 timed, median):
    #   detector      int8 33.2 ms -> fp16  8.2 ms   4.0x
    #   segmentation  int8 50.3 ms -> fp16 31.0 ms   1.6x
    # int8 was also measurably less accurate: confident detections (score > 0.3)
    # agreed with fp16 on only 71.9% of boxes at IoU 0.9 for the detector. fp16
    # costs about 2x the file size (27.7 -> 54.7 MB detector).
    # DEVICE WARNING: use compute_unit=CPU_AND_GPU. Measured on an iPhone 16
    # (iOS 26.6.1) 2026-09-04 by pulling raw .pte outputs off the device and
    # matching detections to an eager reference by IoU. There are two separate
    # defects, and only one of them is about the compute unit:
    #
    # 1. The ANE translates the whole prediction. Mask and box move together,
    #    right and down, by a small constant:
    #      ALL (reaches the ANE)   mask +1.98 px x, +1.09 px y  (312-px input)
    #      CPU_AND_GPU             mask +0.00 px x, -0.04 px y
    #    About 7 px on a phone screen, which is the shift reported in #1406.
    #    IoU barely notices it (0.982 vs 0.996 on a 134x291 box), so measure the
    #    translation directly by aligning the masks rather than trusting IoU.
    #
    # 2. The `[0]` output select corrupted Core ML's CPU path - see the wrapper
    #    above. Fixed there with squeeze(0). It was never an ANE problem, and it
    #    is why CPU_ONLY looked like it "destroyed" the geometry while leaving
    #    class and score correct.
    #
    # fp32 on ALL is also correct but costs 2x the file size, so fp16 on
    # CPU_AND_GPU is what ships.
    #
    # Do NOT compare detections by query index. Near-tied queries exchange slots
    # between compute units, and box and mask swap together, so an index-wise
    # diff reports a fictitious 158-px "shift" where nothing moved. Query-slot
    # instability is baseline here: the arms disagree on the class of 57-72 of
    # the 100 low-score queries, CPU_AND_GPU included.
    #
    # A Mac cannot check any of this: its ANE compiler rejects the model and Core
    # ML falls back to CPU/GPU without saying so, which is why every host-side
    # validation passed. Read the numbers off a phone.
    #
    # NOTE for rfdetr_keypoint: its DEVICE WARNING concludes "it is NOT the ANE"
    # because fp16 on CPU_ONLY failed there too. That inference does not hold now
    # that the CPU_ONLY failure is known to be the output select rather than a
    # compute-unit problem. Keypoint was never tried on CPU_AND_GPU and may not
    # need fp32 either; by its own numbers fp16 was 4.0x faster.
    quant = {"mode": "linear_symmetric", "dtype": "int8", "granularity": "per_channel"}
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        # Excludes the ANE. See the DEVICE WARNING above: this is not a tuning
        # knob, the ANE's fp16 path is wrong for this graph.
        compute_unit=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=(
            ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
        ),
        op_linear_quantizer_config=quant if precision == "int8" else None,
    )
    # lower_full_graph=False lets the partitioner leave ops CoreML can't take
    # (e.g. a rank-6 reshape) on the CPU backend instead of failing the export.
    return to_edge_transform_and_lower(
        exported,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=[CoreMLPartitioner(compile_specs=specs, lower_full_graph=False)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()


def export(config: dict, out_dir: str) -> str:
    _apply_patches()
    from rfdetr import RFDETRSegNano

    variant = config["variant"]
    model = RFDETRSegNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    resolution = model.model.resolution

    wrapper = _RFDETRSegmentationWrapper(actual, resolution).eval()
    example = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)
    with torch.inference_mode():
        ep = torch_export(wrapper, args=example, strict=False)

    program = _lower(ep, variant["precision"])
    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
