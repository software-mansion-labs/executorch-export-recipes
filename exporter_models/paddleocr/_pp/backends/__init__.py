"""Backend-specific export machinery for the PaddleOCR exporters.

Every PaddleOCR export script (det/rec, layout, orientation, table, dewarp, fused
helpers) shares the SAME three delegate targets — only the model architecture and
its calibration differ. This package isolates the per-backend "patches" so the
export scripts stay pure model logic:

  backends.xnnpack  CPU  — XnnpackPartitioner, the post-permute contiguous pass,
                           the PT2E static-int8 primitive, the RT-DETR layout
                           partitioner (ViewCopyConfig + int8wo bridge).
  backends.coreml   ANE  — compile-spec / partitioner builder (weight-only int8,
                           enumerated shapes, COMPILED_MODEL), layout variant.
  backends.vulkan   GPU  — VulkanPartitioner (fp16, optional dynamic-resize) and
                           the opt-in upsample-resize patch.

Lowering/serialization boilerplate that is backend-agnostic lives one level up in
`lowering.py`; the HF-output wrappers in `wrappers.py`; calibration image loaders
in `calib.py`.

Heavy backend imports (executorch.backends.*, coremltools) are done lazily inside
the builder functions so importing this package — or running `--help` on a script
— never drags in a delegate toolchain that isn't needed.
"""
