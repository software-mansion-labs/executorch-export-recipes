"""Vulkan (Android GPU) backend helpers for the EasyOCR exporter.

True dynamic via runtime tensor resize. fp16 only (int8/scan can't resize). Canonical
fp16 = fp32 graph + force_fp16, storing real half-size weights (unlike XNNPACK's compute-only fp16).

  partitioner(dynamic)     VulkanPartitioner({require_dynamic_shapes, force_fp16}).
  patch_upsample_resize()  opt-in: let CRAFT's bilinear upsamples delegate under dynamic.
"""


def patch_upsample_resize():
    """Flip upsample supports_resize True so CRAFT's scale_factor=2 upsamples delegate under
    dynamic. Scale-path only (output_size None). Opt-in (Meta defaults False; on-device can throw)."""
    import executorch.backends.vulkan.op_registry as R
    def scale_path_only(node):
        return (len(node.args) <= 1) or (node.args[1] is None)
    for k, feat in R.vulkan_supported_ops.items():
        if "upsample" in str(k):
            feat.supports_resize = True
            feat.are_node_inputs_supported_fn = scale_path_only


def partitioner(dynamic, texture_limits=(2048, 2048, 2048), force_fp16=True):
    """Cap texture extents at the 3D-image limit (2048): runtime stores every tensor as a
    VK_IMAGE_TYPE_3D image, so large extents crash vkCreateImage on strict GPUs (Xclipse 940).
    force_fp16=False for an int8 (per-channel) graph — q8ta has no fp16 quantize shader."""
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    return VulkanPartitioner({"require_dynamic_shapes": dynamic, "force_fp16": force_fp16,
                              "texture_limits": texture_limits})
