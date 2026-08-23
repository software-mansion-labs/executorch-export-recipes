"""CoreML (Apple ANE/GPU) backend helpers. No RangeDim -> enumerated shapes (fixed set
compiled once). Precision = weight-only int8 (fp16 compute on ANE, immune to calib drift).
COMPILED_MODEL bakes the .mlmodelc into the PTE to kill the enumerated cold start."""


def _qcfg(quantize):
    """op_linear_quantizer_config for weight-only int8/int4 (None = no quant)."""
    if quantize == "int8":
        return {"mode": "linear_symmetric", "dtype": "int8",
                "granularity": "per_channel", "weight_threshold": 512}
    if quantize == "int4":
        return {"mode": "linear_symmetric", "dtype": "int4",
                "granularity": "per_block", "block_size": 32, "weight_threshold": 512}
    return None


def compile_specs(precision, *, quantize=None, compute_unit="ALL", compiled_model=True,
                  deployment="iOS18"):
    """Base compile-spec list. precision 'fp32'/'float32' -> FLOAT32, else FLOAT16."""
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    cprec = ct.precision.FLOAT32 if precision in ("fp32", "float32") else ct.precision.FLOAT16
    cu = ct.ComputeUnit.CPU_AND_NE if compute_unit == "CPU_AND_NE" else ct.ComputeUnit.ALL
    kw = dict(compute_precision=cprec, compute_unit=cu,
              minimum_deployment_target=getattr(ct.target, deployment),
              op_linear_quantizer_config=_qcfg(quantize))
    if compiled_model:
        kw["model_type"] = CoreMLBackend.MODEL_TYPE.COMPILED_MODEL
    return list(CoreMLBackend.generate_compile_specs(**kw))


def partitioner(precision, *, quantize=None, lower_full_graph=False, compute_unit="ALL",
                compiled_model=True, extra_specs=None):
    """CoreMLPartitioner. `extra_specs` appends e.g. an enumerated-shapes spec;
    `lower_full_graph=True` is required for enumerated."""
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    specs = compile_specs(precision, quantize=quantize, compute_unit=compute_unit,
                          compiled_model=compiled_model)
    if extra_specs:
        specs = specs + list(extra_specs)
    return CoreMLPartitioner(compile_specs=specs, lower_full_graph=lower_full_graph)


def enumerated_spec(ep, enum_map_for_method):
    """generate_enumerated_shapes_compile_spec(ep, {in_name: [[1,c,h,w],...]})."""
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    return CoreMLBackend.generate_enumerated_shapes_compile_spec(ep, enum_map_for_method)


def guard_fuse_linear_bias():
    """Swallow coremltools fuse_linear_bias IndexError/ValueError (RT-DETR layout path)."""
    try:
        from coremltools.converters.mil.mil.passes.defs import optimize_linear as _ol
        _orig = _ol.fuse_linear_bias.apply
        def _safe(self, prog):
            try:
                return _orig(self, prog)
            except (IndexError, ValueError):
                pass
        _ol.fuse_linear_bias.apply = _safe
    except Exception:
        pass


def layout_partitioner(precision, quantize, compiled_model=True):
    """RT-DETR (PP-DocLayoutV3): CPU_AND_NE, weight-only int8/int4 appended as a separate
    op-quantizer spec; fuse_linear_bias guarded. compiled_model bakes .mlmodelc (avoids ~30s
    cold start). precision: 'float32'/'float16'. quantize: 'none'/'int8'/'int4'."""
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    guard_fuse_linear_bias()
    prec = ct.precision.FLOAT32 if precision == "float32" else ct.precision.FLOAT16
    spec_kw = dict(compute_unit=ct.ComputeUnit.CPU_AND_NE,
                   minimum_deployment_target=ct.target.iOS18, compute_precision=prec)
    if compiled_model:
        spec_kw["model_type"] = CoreMLBackend.MODEL_TYPE.COMPILED_MODEL
    specs = CoreMLBackend.generate_compile_specs(**spec_kw)
    if quantize == "int8":
        specs.append(CoreMLBackend.generate_op_linear_quantizer_config_compile_spec({
            "mode": "linear_symmetric", "dtype": "int8",
            "granularity": "per_channel", "weight_threshold": 512}))
    elif quantize == "int4":
        specs.append(CoreMLBackend.generate_op_linear_quantizer_config_compile_spec({
            "mode": "linear_symmetric", "dtype": "int4",
            "granularity": "per_block", "block_size": 32, "weight_threshold": 512}))
    return CoreMLPartitioner(compile_specs=specs)
