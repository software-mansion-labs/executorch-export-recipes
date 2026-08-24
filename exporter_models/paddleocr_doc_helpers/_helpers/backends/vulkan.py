"""Vulkan (Android GPU) backend helpers. True dynamic via runtime resize. fp16/fp32 only
(int8 can't resize); real half-size fp16 weights. Canonical fp16 = fp32 graph + force_fp16
in the partitioner — never a hand-rolled .half() graph (crashes the delegate on device)."""


def patch_upsample_resize():
    """Flip Vulkan upsample supports_resize -> True so FPN bilinear upsamples delegate
    under dynamic (else portable CPU, 4 blobs). Only the scale_factors path (output_size
    None) is allowed. Opt-in: forcing it crashed on-device (Xclipse 940)."""
    import executorch.backends.vulkan.op_registry as R
    def scale_path_only(node):
        return (len(node.args) <= 1) or (node.args[1] is None)   # output_size None
    for k, feat in R.vulkan_supported_ops.items():
        if "upsample" in str(k):
            feat.supports_resize = True
            feat.are_node_inputs_supported_fn = scale_path_only


def partitioner(*, dynamic=False, force_fp16=True, plain=False, texture_limits=None,
                nn_module_allowlist=None, nn_module_blocklist=None):
    """VulkanPartitioner. `plain=True` returns bare defaults (RT-DETR layout); otherwise
    {require_dynamic_shapes: dynamic, force_fp16}.

    nn_module_allowlist/blocklist: nn.Module typename substrings gating which nodes
    delegate. RT-DETR layout uses ["PPDocLayoutV3Model"] to force the output-assembly
    ops onto CPU (a multi-output Vulkan delegate mis-binds outputs on-device).

    texture_limits: (x,y,z) cap on 3D-texture extents. Default (16384,16384,2048) lets a
    width-packed tensor exceed maxImageDimension3D (2048) -> vkCreateImage crash on strict
    GPUs (Xclipse 940). SVTR vocab V=18710 -> X-extent 4678 > 2048; (2048,2048,2048) forces
    it to buffer/portable. Only needed for strict GPUs."""
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    if plain:
        return VulkanPartitioner()
    opts = {"require_dynamic_shapes": dynamic, "force_fp16": force_fp16}
    if texture_limits is not None:
        opts["texture_limits"] = texture_limits
    kw = {}
    if nn_module_allowlist is not None:
        kw["nn_module_allowlist"] = list(nn_module_allowlist)
    if nn_module_blocklist is not None:
        kw["nn_module_blocklist"] = list(nn_module_blocklist)
    return VulkanPartitioner(opts, **kw)
