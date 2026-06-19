# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CLI orchestrator for ExecuTorch export with XNNPACK delegation.

Run with:

    uv run src/rfdetr/export/executorch_main.py \\
        --variant RFDETRKeypointPreview \\
        --pretrain-weights path/to/checkpoint.pth \\
        --output-dir output

Produces ``<output_dir>/<variant_size>.pte`` ready for the ExecuTorch runtime.
"""

from __future__ import annotations

import argparse
import importlib
from copy import deepcopy

import torch
import torch.nn as nn

# Bundled alongside this script (a copy of the model repo's export patches) so the
# export logic is self-contained here; only the model classes come from `rfdetr`.
from export_patches import apply_patches
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class RFDETRKeypointWrapper(nn.Module):
    """Conforms LWDETR keypoint outputs to the on-device contract:

    boxes     (Q, 4)    xyxy bbox in model input pixel space
    scores    (Q,)      person confidence in [0, 1]
    keypoints (Q, K, 3) per-detection (x, y, visibility); xy in pixel space
    """

    def __init__(self, model: nn.Module, resolution: int, person_class_index: int = 1, num_keypoints: int = 17):
        super().__init__()
        self.model = model
        self.resolution = resolution
        self.person_class_index = person_class_index
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor):
        # Call forward_export directly rather than self.model(x): LWDETR.export() swaps
        # forward to an *instance* attribute, which makes torch.export's run_decompositions
        # mis-resolve lifted constants on quantized graphs ("model.forward.__self__…").
        # forward_export is the same export-mode forward.
        forward_fn = getattr(self.model, "forward_export", self.model)
        pred_boxes, pred_logits, pred_keypoints = forward_fn(x)

        # boxes: (1, Q, 4) cxcywh in [0,1] → (Q, 4) xyxy in pixel space
        cxy = pred_boxes[0, :, :2]
        wh = pred_boxes[0, :, 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        boxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution

        # scores: (1, Q, C) logits → (Q,) sigmoid person-class confidence
        scores = pred_logits[0, :, self.person_class_index].sigmoid()

        # keypoints: (1, Q, num_classes*K, 8) → person-class slice → (Q, K, 3)
        # Channel layout per keypoint (8 floats):
        #   [0:2]  x, y in normalized [0, 1] coords
        #   [2]    "findable" logit — sigmoid = P(keypoint is annotated in COCO sense)
        #   [3]    "visible"  logit — sigmoid = P(keypoint is annotated AND unoccluded)
        #   [4:7]  uncertainty / sigma predictions (location confidence)
        #   [7]    class-logit contribution (used internally by the detection head;
        #          NOT a per-keypoint visibility — sigmoid stays high for absent kps)
        # Channel 2 (findable) matches the canonical postprocess in src/rfdetr/models/postprocess.py.
        kp_start = self.person_class_index * self.num_keypoints
        kp_slice = pred_keypoints[0, :, kp_start : kp_start + self.num_keypoints, :]
        # The model emits free-regression xy; nothing keeps it inside [0, resolution].
        # Clamp to model-input bounds so downstream scaling can't render off-screen.
        kp_xy = (kp_slice[..., :2] * self.resolution).clamp(min=0.0, max=float(self.resolution))
        kp_vis = kp_slice[..., 2].sigmoid().unsqueeze(-1)
        keypoints = torch.cat([kp_xy, kp_vis], dim=-1)

        return boxes, scores, keypoints


class RFDETRDetectionWrapper(nn.Module):
    """Plain detection forward_export → (boxes, logits)."""

    def __init__(self, model: nn.Module, resolution: int):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def forward(self, x: torch.Tensor):
        forward_fn = getattr(self.model, "forward_export", self.model)
        pred_boxes, pred_logits = forward_fn(x)
        scores, labels = pred_logits[0].sigmoid().max(dim=-1)
        boxes = pred_boxes[0]
        cxy = boxes[..., :2]
        wh = boxes[..., 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        bboxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution
        return bboxes, scores, labels.float()


def _print_delegation(program) -> None:
    from executorch.devtools.backend_debug import get_delegation_info
    from tabulate import tabulate

    graph_module = program.exported_program().graph_module
    delegation_info = get_delegation_info(graph_module)
    logger.info(delegation_info.get_summary())
    df = delegation_info.get_operator_delegation_dataframe()
    logger.info("\n" + tabulate(df, headers="keys", tablefmt="fancy_grid"))


def _lower_xnnpack(exported, out_path: str) -> None:
    """Lower an ExportedProgram to a .pte buffer via XNNPACK (CPU; ARM/x86)."""
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    program = to_edge_transform_and_lower(
        exported,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
    ).to_executorch()

    _print_delegation(program)
    with open(out_path, "wb") as f:
        f.write(program.buffer)
    logger.info(f"Saved {out_path}")


def _export_xnnpack_quantized(module: nn.Module, example: tuple, dynamic: bool = True):
    """PT2E int8 quantize ``module`` for XNNPACK and return an ``ExportedProgram``.

    Uses the XNNPACK symmetric quantizer (per-channel int8 weights). ``dynamic``
    selects dynamic activation quantization (no calibration data needed; good for
    the linear-heavy transformer); ``dynamic=False`` is static int8 calibrated on
    the single example input.

    NOTE: XNNPACK is a CPU backend. On Apple Silicon it cannot match the CoreML
    ANE/GPU delegate — this path exists for portability (non-Apple ARM/x86) and to
    quantify the CPU quantization tradeoff, not as the fastest option on a Mac.

    KNOWN LIMITATION (RF-DETR keypoint model): the calibration forward raises
    "Tensor dtype mismatch! Expected: Long, Got: float" because PT2E inserts
    activation observers on the integer index tensors feeding the model's
    topk/gather postprocess. Avoiding this needs a module-scoped quantizer that
    only quantizes the transformer's nn.Linear blocks; not pursued here because a
    CPU int8 build is ~7x slower than the CoreML delegate on Apple Silicon anyway.

    Args:
        module: The eval-mode wrapper module to quantize.
        example: Example positional inputs ``(tensor,)`` for tracing/calibration.
        dynamic: Dynamic (True) vs static (False) activation quantization.

    Returns:
        The quantized ``ExportedProgram`` ready for XNNPACK lowering.
    """
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    # Capture an ATen GraphModule to feed PT2E. export_for_training is the modern
    # capture entry point; fall back to plain export().module() on older builds.
    try:
        captured = torch.export.export_for_training(module, example, strict=True).module()
    except AttributeError:
        captured = torch.export.export(module, example, strict=True).module()

    qconfig = get_symmetric_quantization_config(is_per_channel=True, is_dynamic=dynamic)
    quantizer = XNNPACKQuantizer().set_global(qconfig)
    prepared = prepare_pt2e(captured, quantizer)
    prepared(*example)  # calibration forward (observe activation ranges)
    converted = convert_pt2e(prepared)
    return torch.export.export(converted, example, strict=True)


def _lower_coreml(
    exported,
    out_path: str,
    compute_unit: str = "all",
    quantize: bool = False,
    quant_dtype: str = "int8",
    quant_block_size: int = 0,
    min_deployment_target: str = "default",
    compute_precision: str = "fp32",
    addmm_transform: bool = True,
) -> None:
    """Lower an ExportedProgram to a .pte buffer via CoreML (Apple ANE + GPU + CPU).

    ``quantize=True`` enables symmetric weight-only quantization (``quant_dtype``
    is ``"int8"`` or ``"int4"``). Smaller file, but it does NOT speed up ANE/GPU
    inference (weights are dequantized at runtime; compute stays fp16) and it
    measurably hurts keypoint accuracy. Default fp32.

    Args:
        quant_dtype: ``"int8"`` or ``"int4"`` weight dtype when ``quantize``.
        quant_block_size: If > 0, use per-block granularity with this block size
            (recommended for int4); 0 means per-channel.
    """
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    cu_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    # NOTE: iOS18 (and iOS17) selects newer CoreML op lowerings that BREAK this
    # keypoint model — detections vanish / keypoints drift >100px (worst at non-576
    # resolutions, but degraded even at 576). "default" (None → coremltools' own
    # default, conservative lowering) is both correct AND fastest (~47ms vs iOS16's
    # ~75ms at 576). iOS16 is the correct fallback for low-res exports. See
    # memory/keypoint-perf-quantization-findings.md.
    target_map = {
        "default": None,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "iOS18": ct.target.iOS18,
    }
    precision_map = {"fp32": ct.precision.FLOAT32, "fp16": ct.precision.FLOAT16}
    compile_spec_kwargs = dict(
        compute_unit=cu_map[compute_unit],
        minimum_deployment_target=target_map[min_deployment_target],
        compute_precision=precision_map[compute_precision],
    )
    if quantize:
        quant_cfg = {
            "mode": "linear_symmetric",
            "dtype": quant_dtype,
            "granularity": "per_block" if quant_block_size else "per_channel",
        }
        if quant_block_size:
            quant_cfg["block_size"] = quant_block_size
        compile_spec_kwargs["op_linear_quantizer_config"] = quant_cfg
    compile_specs = CoreMLBackend.generate_compile_specs(**compile_spec_kwargs)
    transform_passes = [AddmmToLinearTransform()] if addmm_transform else []
    program = to_edge_transform_and_lower(
        exported,
        transform_passes=transform_passes,
        partitioner=[CoreMLPartitioner(compile_specs=compile_specs, lower_full_graph=False)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()

    _print_delegation(program)
    with open(out_path, "wb") as f:
        f.write(program.buffer)
    logger.info(f"Saved {out_path}")


def _lower_mlx(exported, out_path: str) -> None:
    """Lower an ExportedProgram to a ``.pte`` buffer via the MLX delegate (Apple GPU).

    MLX is an experimental ExecuTorch backend that runs delegated subgraphs on the
    Apple Silicon GPU through the MLX framework. It is only available when ExecuTorch
    is built from source with the MLX backend (auto-included on Apple Silicon when the
    Metal compiler from Xcode is present).

    Args:
        exported: The traced ``ExportedProgram`` to lower.
        out_path: Filesystem path the serialized ``.pte`` buffer is written to.
    """
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.capture._config import ExecutorchBackendConfig

    # Do NOT call exported.run_decompositions() here: to_edge_transform_and_lower
    # decomposes internally (this matches the MLX backend's own test harness), and an
    # explicit run_decompositions lowers torchao-quantized linears through aten.t,
    # which the int-weight tensor subclass can't trace.
    #
    # _check_ir_validity=False / _skip_dim_order=True mirror the MLX LLM examples: the
    # default IR-validity + dim-order path trips a graph-lint on torchao-quantized graphs
    # (lifted_tensor_0). Harmless for the unquantized fp32 path; MLX manages memory order.
    program = to_edge_transform_and_lower(
        exported,
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    ).to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True),
    )

    _print_delegation(program)
    with open(out_path, "wb") as f:
        f.write(program.buffer)
    logger.info(f"Saved {out_path}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Export an RF-DETR model to ExecuTorch (.pte) with XNNPACK.")
    parser.add_argument(
        "--variant",
        default="RFDETRKeypointPreview",
        help="RFDETR variant class name from rfdetr.variants (e.g. RFDETRNano, RFDETRKeypointPreview).",
    )
    parser.add_argument("--pretrain-weights", default=None, help="Path to checkpoint .pth to load.")
    parser.add_argument("--output-dir", default="output", help="Directory to write the exported .pte to.")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output .pte filename without extension. Overrides the auto-derived name "
        "(e.g. --output-name rfdetr_keypoint_preview_mlx_fp32).",
    )
    parser.add_argument("--shape", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override the model's square input resolution (must be divisible by 24). "
        "Lower resolution = quadratically less compute = faster, at some accuracy cost. "
        "Re-interpolates positional encodings; the example input follows this automatically.",
    )
    parser.add_argument(
        "--backend",
        default="xnnpack",
        choices=["xnnpack", "coreml", "mlx"],
        help=(
            "Lowering backend. xnnpack: portable CPU. coreml: Apple ANE/GPU/CPU (int8-weight quantized). "
            "mlx: experimental Apple-GPU delegate (requires ExecuTorch built from source with the MLX backend)."
        ),
    )
    parser.add_argument(
        "--compute-unit",
        default="all",
        choices=["all", "cpu_and_ne", "cpu_and_gpu", "cpu_only"],
        help="CoreML compute unit (ignored for xnnpack). cpu_and_ne pins to the ANE, "
        "cpu_and_gpu pins to the GPU — pair them to run two tiles concurrently on different engines.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply int8 per-channel weight quantization (CoreML only; alias for --coreml-quant int8). "
        "Smaller file, but lower accuracy and NOT faster on ANE/GPU.",
    )
    parser.add_argument(
        "--coreml-quant",
        default=None,
        choices=["int8", "int4"],
        help="CoreML weight-only quantization dtype. int4 ~8x smaller than fp32. "
        "Weights are dequantized at runtime, so this shrinks the file but does not speed up ANE/GPU compute.",
    )
    parser.add_argument(
        "--coreml-quant-block-size",
        type=int,
        default=0,
        help="Per-block granularity block size for CoreML quantization (0 = per-channel). "
        "Recommended for int4 (e.g. 32) to limit accuracy loss.",
    )
    parser.add_argument(
        "--mlx-qlinear",
        default="none",
        choices=["none", "4w", "8w"],
        help="MLX weight-only Linear quantization: 4w=int4, 8w=int8 (MLX backend only). "
        "torchao source-transform applied before export; the MLX delegate fuses dequantize+linear. "
        "Activations stay fp32, so coordinate outputs keep full precision. Skips shape-incompatible layers.",
    )
    parser.add_argument(
        "--mlx-group-size",
        type=int,
        default=64,
        help="Group size for MLX weight quantization (4w sweet spot: 64).",
    )
    parser.add_argument(
        "--xnnpack-quant",
        default="none",
        choices=["none", "dynamic", "static"],
        help="XNNPACK int8 PT2E quantization: dynamic (no calibration) or static (calibrated on the "
        "example input). CPU backend — for portability, not the fastest option on Apple Silicon.",
    )
    parser.add_argument(
        "--min-deployment-target",
        default="default",
        choices=["default", "iOS16", "iOS17", "iOS18"],
        help="CoreML minimum deployment target. 'default' (coremltools default, conservative "
        "lowerings) is correct AND fastest for this model; iOS17/iOS18 BREAK the keypoint head. "
        "Use iOS16 for low-resolution exports.",
    )
    parser.add_argument(
        "--compute-precision",
        default="fp32",
        choices=["fp32", "fp16"],
        help="CoreML compute precision. Note: ANE is fp16-only hardware; fp32 may be silently cast.",
    )
    parser.add_argument(
        "--no-addmm-transform",
        action="store_true",
        help="Skip AddmmToLinearTransform. Sometimes helps when CoreML iOS mis-handles fused linear weights.",
    )
    # Patch toggles — start minimal and escalate.
    parser.add_argument("--patch-transformer-forward", action="store_true")
    parser.add_argument("--patch-decoder-forward", action="store_true")
    parser.add_argument("--patch-decoder-layer-forward", action="store_true")
    args = parser.parse_args()

    # --quantize is a back-compat alias for --coreml-quant int8.
    if args.quantize and args.coreml_quant is None:
        args.coreml_quant = "int8"

    # CoreML caps tensor rank at 5; upstream MSDeformAttn produces a rank-6
    # sampling_locations. The patched ms_deform_attn collapses (n_levels, n_points)
    # into one axis so it stays rank-5 — required for CoreML, harmless for XNNPACK.
    needs_rank5_msdeform = args.backend == "coreml"
    # MLX has no aten.grid_sampler_2d handler, and its getitem handler can't index
    # the tuple Tensor.split returns. The grid-free deform core swaps grid_sample
    # for gather-based sampling and split[idx] for narrow(), so it lowers to MLX.
    needs_grid_free_msdeform = args.backend == "mlx"
    apply_patches(
        proposals=False,
        ms_deform_attn_forward=needs_rank5_msdeform,
        ms_deform_attn_core=needs_rank5_msdeform,
        ms_deform_attn_grid_free=needs_grid_free_msdeform,
        pos_embed_compile_safe=needs_grid_free_msdeform,
        transformer_forward=args.patch_transformer_forward,
        decoder_forward=args.patch_decoder_forward,
        decoder_layer_forward=args.patch_decoder_layer_forward,
    )

    variants = importlib.import_module("rfdetr.variants")
    if not hasattr(variants, args.variant):
        raise SystemExit(f"Unknown variant {args.variant!r}. See rfdetr.variants for available classes.")
    model_cls = getattr(variants, args.variant)

    model_kwargs = {}
    if args.pretrain_weights is not None:
        model_kwargs["pretrain_weights"] = args.pretrain_weights
    if args.resolution is not None:
        model_kwargs["resolution"] = args.resolution
    wrapper_meta = model_cls(**model_kwargs)

    inner = deepcopy(wrapper_meta.model.model.to("cpu")).eval()
    inner.export()
    # LWDETR.export() sets forward to an instance attribute (self.forward = self.forward_export);
    # the wrappers call model.forward_export(x) directly, so restore the original forward to keep
    # `inner` a plain module (the instance swap breaks run_decompositions on quantized graphs).
    if hasattr(inner, "_forward_origin"):
        inner.forward = inner._forward_origin

    if args.backend == "mlx" and args.mlx_qlinear != "none":
        from torchao.quantization import quantize_
        from torchao.quantization.granularity import PerAxis, PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig

        # torchao.quantize_ mutates weights in-place, which fails on inference-mode
        # tensors ("Cannot set version_counter for inference tensor"). The checkpoint
        # loads weights under inference mode, so re-materialize params/buffers as
        # normal tensors before quantizing.
        with torch.inference_mode(False), torch.no_grad():
            for m in inner.modules():
                for n, p in list(m._parameters.items()):
                    if p is not None:
                        m._parameters[n] = torch.nn.Parameter(p.clone(), requires_grad=False)
                for n, b in list(m._buffers.items()):
                    if b is not None and torch.is_tensor(b):
                        m._buffers[n] = b.clone()

        # Weight-only int quantization (no packing -> no CUDA; activations stay fp32, so
        # coordinate outputs keep full precision). SKIP linears inside nn.MultiheadAttention:
        # multi_head_attention_forward calls aten.t on the weight, which the int-weight
        # tensor subclass can't trace. The MLX delegate fuses the dequantize+linear pattern.
        gs = args.mlx_group_size
        weight_dtype = torch.int4 if args.mlx_qlinear == "4w" else torch.int8
        mha_linear_ids = {
            id(sub)
            for mod in inner.modules()
            if isinstance(mod, torch.nn.MultiheadAttention)
            for sub in mod.modules()
            if isinstance(sub, torch.nn.Linear)
        }

        def _qfilter(m, fqn):
            if not isinstance(m, torch.nn.Linear) or id(m) in mha_linear_ids:
                return False
            return (m.weight.shape[-1] % gs == 0) if gs else True

        config = IntxWeightOnlyConfig(
            weight_dtype=weight_dtype,
            granularity=PerAxis(0) if gs == 0 else PerGroup(gs),
            intx_choose_qparams_algorithm="hqq_scale_only",
        )
        logger.info(
            f"Applying MLX {args.mlx_qlinear} weight-only quantization "
            f"(group_size={gs}, skipping MultiheadAttention linears)…"
        )
        quantize_(inner, config, filter_fn=_qfilter)

        # torchao's quantized subclasses otherwise try to re-wrap their int inner tensors
        # as grad-enabled Parameters, which mis-lifts under strict export (graph-lint
        # "references nonexistent attribute lifted_tensor_0"). Disable grad everywhere.
        for p in inner.parameters():
            p.requires_grad_(False)

    resolution = wrapper_meta.model.resolution
    shape = tuple(args.shape) if args.shape is not None else (resolution, resolution)

    is_keypoint = (
        bool(getattr(wrapper_meta.model_config, "use_grouppose_keypoints", False)) or "Keypoint" in args.variant
    )
    if is_keypoint:
        # num_keypoints_per_class is [0, 17] for COCO person — pick the first class with kps.
        kps_per_class = getattr(inner, "num_keypoints_per_class", [0, 17]) or [0, 17]
        person_class_index = next((i for i, n in enumerate(kps_per_class) if n > 0), 1)
        num_keypoints = kps_per_class[person_class_index]
        module = RFDETRKeypointWrapper(inner, resolution, person_class_index, num_keypoints).eval()
    else:
        module = RFDETRDetectionWrapper(inner, resolution).eval()

    num_channels = getattr(wrapper_meta.model_config, "num_channels", 3)
    example = (torch.randn(1, num_channels, shape[0], shape[1]),)

    import os

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = getattr(wrapper_meta, "size", "rfdetr")
    suffix = args.backend
    if args.backend == "coreml":
        suffix = f"coreml-{args.min_deployment_target}-{args.compute_unit}-{args.compute_precision}"
        if args.coreml_quant:
            suffix += f"-{args.coreml_quant}"
            if args.coreml_quant_block_size:
                suffix += f"-bs{args.coreml_quant_block_size}"
        if args.no_addmm_transform:
            suffix += "-noaddmm"
    elif args.backend == "mlx" and args.mlx_qlinear != "none":
        suffix = f"mlx-{args.mlx_qlinear}-gs{args.mlx_group_size}"
    elif args.backend == "xnnpack" and args.xnnpack_quant != "none":
        suffix = f"xnnpack-int8-{args.xnnpack_quant}"
    if args.output_name:
        out_path = os.path.join(args.output_dir, f"{args.output_name}.pte")
    else:
        out_path = os.path.join(args.output_dir, f"{base_name}-{suffix}.pte")

    is_quantized_mlx = args.backend == "mlx" and args.mlx_qlinear != "none"
    is_quantized_xnnpack = args.backend == "xnnpack" and args.xnnpack_quant != "none"
    # torchao-quantized graphs only trace correctly under strict (dynamo) export —
    # non-strict make_fx lowers the int-weight subclass through aten.t, which it can't
    # handle. They also can't trace under inference_mode (its make_fx tensors hit
    # version_counter), so use no_grad. Unquantized stays strict=False + inference_mode.
    strict = is_quantized_mlx
    trace_ctx = torch.no_grad() if is_quantized_mlx else torch.inference_mode()
    if is_quantized_xnnpack:
        logger.info(f"PT2E int8 quantizing for XNNPACK ({args.xnnpack_quant})…")
        with torch.no_grad():
            exported = _export_xnnpack_quantized(module, example, dynamic=args.xnnpack_quant == "dynamic")
    else:
        logger.info(f"Tracing model with torch.export (strict={strict})…")
        with trace_ctx:
            exported = torch.export.export(module, args=example, strict=strict)

    if args.backend == "xnnpack":
        logger.info("Lowering to ExecuTorch + XNNPACK…")
        _lower_xnnpack(exported, out_path)
    elif args.backend == "mlx":
        logger.info("Lowering to ExecuTorch + MLX (experimental Apple-GPU delegate)…")
        _lower_mlx(exported, out_path)
    else:
        logger.info(
            f"Lowering to ExecuTorch + CoreML (compute_unit={args.compute_unit}, "
            f"quant={args.coreml_quant}, target={args.min_deployment_target}, "
            f"precision={args.compute_precision})…"
        )
        _lower_coreml(
            exported,
            out_path,
            compute_unit=args.compute_unit,
            quantize=args.coreml_quant is not None,
            quant_dtype=args.coreml_quant or "int8",
            quant_block_size=args.coreml_quant_block_size,
            min_deployment_target=args.min_deployment_target,
            compute_precision=args.compute_precision,
            addmm_transform=not args.no_addmm_transform,
        )


if __name__ == "__main__":
    _cli()
