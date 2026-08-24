---
name: vulkan-export
description: Vulkan backend knowledge for exporting ExecuTorch models — inline VulkanPartitioner lowering, fp16 compute, the 3D-texture limit, dynamic-via-resize, and mixed-delegate fusion.
---

# Vulkan Export

Vulkan is the Android **GPU** backend for ExecuTorch. It ships inside executorch
(like XNNPACK) — no pip extra. There is **no shared `lower()` helper**; Vulkan
models lower inline in their `export.py` (same as CoreML/MLX).

## Lowering

    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import to_edge_transform_and_lower
    to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner(compile_options)]).to_executorch()

Compute is **fp16** on the GPU regardless of the graph dtype (export an fp32
graph + `force_fp16`); the deliverable is effectively half-size fp16. fp16 is a
real size lever here (unlike XNNPACK, where it is compute-only).

## The 3D-texture limit (the #1 Vulkan crash)

Vulkan stores tensors as 3D textures. A tensor whose any dim exceeds the GPU's
`maxImageDimension3D` (commonly 2048) makes `vkCreateImage` fail at runtime
(Xclipse: a HostFunction crash). The classic trigger is a **large vocab
projection** (PP-OCR SVTR V=18710, multilingual recognizers). Cap it at export:

    VulkanPartitioner(compile_options={"texture_limits": (2048, 2048, 2048)})

so oversized tensors fall back instead of crashing on device. Parse the PTE
delegate flatbuffer offline to catch oversized 3D-texture tensors before shipping.

## Dynamic shapes

Vulkan does **true dynamic** via runtime resize (`require_dynamic_shapes`), fp16
only — int8 (q8ta), upsample, and scan ops do **not** resize. For OCR we ship
**static bucketed** methods instead (fully delegated, no resize footguns).

## Mixed-delegate fusion (one PTE, two backends)

Some ops crash or won't delegate on Vulkan but run fine on CPU. You can put
different methods of one fused PTE on different delegates by passing a
**per-method partitioner dict** to `to_edge_transform_and_lower`:

    parts = {m: [VulkanPartitioner(...)] for m in det_methods}
    parts.update({m: [XnnpackPartitioner()] for m in rec_methods})
    to_edge_transform_and_lower(eps, partitioner=parts)

PP-OCR ships exactly this: detect on Vulkan (fp16), recognize on XNNPACK (the
SVTR vocab head crashes Vulkan's `make_ivec2`). EasyOCR's CRNN BiLSTM also won't
delegate to Vulkan, so its recognize runs on XNNPACK too.

## Usage in export.py

Lower inline; never call the shared `exporter.backends.lower` (it is xnnpack-only
and raises for other backends). Heavy imports stay inside `export()`.
