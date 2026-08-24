"""CoreML (Apple ANE/GPU) backend helpers for the EasyOCR exporter.

No true dynamic -> enumerated (CRAFT) or static (CRNN: scan can't enumerate).
Weight-only int8 (fp16 compute on ANE); COMPILED_MODEL bakes the .mlmodelc into the PTE.

  compile_specs(...)       base spec list (precision, weight-only quant, iOS18).
  partitioner(...)         CoreMLPartitioner; extra_specs appends per-method enum spec.
  enumerated_spec(ep, m)   generate_enumerated_shapes_compile_spec for one method.
"""


def _qcfg(quantize):
    if quantize == "int8":
        return {"mode": "linear_symmetric", "dtype": "int8",
                "granularity": "per_channel", "weight_threshold": 512}
    return None


def compile_specs(precision, *, quantize=None, deployment="iOS18"):
    """Base compile-spec list. fp32/float32 -> FLOAT32, else FLOAT16 (ANE fast path)."""
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    cprec = ct.precision.FLOAT32 if precision in ("fp32", "float32") else ct.precision.FLOAT16
    return list(CoreMLBackend.generate_compile_specs(
        compute_precision=cprec, compute_unit=ct.ComputeUnit.ALL,
        model_type=CoreMLBackend.MODEL_TYPE.COMPILED_MODEL,
        minimum_deployment_target=getattr(ct.target, deployment),
        op_linear_quantizer_config=_qcfg(quantize)))


def partitioner(precision, *, quantize=None, lower_full_graph=False, extra_specs=None):
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    specs = compile_specs(precision, quantize=quantize)
    if extra_specs:
        specs = specs + list(extra_specs)
    return CoreMLPartitioner(compile_specs=specs, lower_full_graph=lower_full_graph)


def enumerated_spec(ep, enum_map_for_method):
    """generate_enumerated_shapes_compile_spec(ep, {in_name: [[1,c,h,w],...]})."""
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    return CoreMLBackend.generate_enumerated_shapes_compile_spec(ep, enum_map_for_method)
