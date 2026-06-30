"""Backend-specific export machinery for the EasyOCR exporter (CRAFT + CRNN).

Isolates the per-delegate "patches" so export_easyocr.py stays model/shape logic:

  backends.xnnpack  CPU  — XnnpackPartitioner (MaxDim-stripped + ViewCopyConfig),
                           the post-permute contiguous pass, the PT2E int8 primitive.
  backends.coreml   ANE  — compile-spec / partitioner builder (weight-only int8,
                           enumerated shapes, COMPILED_MODEL).
  backends.vulkan   GPU  — VulkanPartitioner (fp16, dynamic-resize) + the opt-in
                           upsample-resize patch (CRAFT 3 bilinear upsamples).

This package is self-contained to the EasyOCR module — it deliberately does NOT
share code with the PaddleOCR exporter's backends package. Heavy backend imports
are lazy (inside the builders) so `--help` never drags in a delegate toolchain.
"""
