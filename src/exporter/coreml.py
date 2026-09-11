"""Core ML lowering helpers shared by the model exporters."""

import inspect
from contextlib import contextmanager


def _backend():
    from executorch.backends.apple.coreml.compiler import CoreMLBackend

    return CoreMLBackend


def supports_embedding_table_quantization() -> bool:
    """Whether the installed ExecuTorch takes the opt-in as a compile spec."""
    params = inspect.signature(_backend().generate_compile_specs).parameters
    return "quantize_embedding_tables" in params


def compile_specs(*, quantize_embedding_tables: bool = False, **kwargs):
    """
    `CoreMLBackend.generate_compile_specs`, plus the embedding-table opt-in.

    On an ExecuTorch that understands the opt-in the flag travels as a compile
    spec. On one that does not, the lowering itself has to be wrapped in
    `quantized_embedding_tables()`; this call cannot do it, because the config
    the flag affects is built inside `preprocess`, long after the specs are.
    """
    if supports_embedding_table_quantization():
        kwargs["quantize_embedding_tables"] = quantize_embedding_tables
    return _backend().generate_compile_specs(**kwargs)


@contextmanager
def quantized_embedding_tables(enabled: bool = True):
    """
    Let `op_linear_quantizer_config` compress embedding tables inside this block.

    ExecuTorch applies that config with `op_type_configs={"gather": None}`, which
    exempts the embedding table. For a transformer that is the right default: the
    body is most of the weight. For a sentence embedder it is backwards, because
    the table IS the model — exempting it leaves paraphrase-multilingual-L12 at
    214 MB instead of 118 MB, for no measured gain in either latency or output
    quality.

    Upstream takes this as a compile spec once pytorch/executorch#22728 lands, and
    then this becomes a no-op. Until it does, the exemption is stripped by patching
    the `OptimizationConfig` the backend constructs. The patch asserts that it
    actually took, so an ExecuTorch that moves this code fails the export rather
    than silently shipping models twice the intended size.
    """
    if not enabled or supports_embedding_table_quantization():
        yield
        return

    import coremltools.optimize as cto
    import executorch.backends.apple.coreml.compiler.coreml_preprocess as pre

    if pre.cto.coreml.OptimizationConfig is not cto.coreml.OptimizationConfig:
        raise RuntimeError(
            "executorch's coreml_preprocess no longer reads OptimizationConfig "
            "from coremltools.optimize; the embedding-table opt-in needs updating"
        )

    original = cto.coreml.OptimizationConfig
    applied = []

    def without_gather_exemption(*args, **kwargs):
        if "op_type_configs" in kwargs or "op_name_configs" in kwargs:
            applied.append(True)
        kwargs.pop("op_type_configs", None)
        kwargs.pop("op_name_configs", None)
        return original(*args, **kwargs)

    pre.cto.coreml.OptimizationConfig = without_gather_exemption
    try:
        yield
    finally:
        pre.cto.coreml.OptimizationConfig = original

    if not applied:
        raise RuntimeError(
            "the gather exemption was never constructed, so quantized_embedding_tables "
            "had nothing to strip: either no quantization ran, or executorch changed "
            "how it configures the weight compressor"
        )
