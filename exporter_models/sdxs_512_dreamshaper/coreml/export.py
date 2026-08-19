import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_pipeline():
    path = _MODEL_DIR / "_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sdxs_512_dreamshaper_coreml_pipeline", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sdxs_512_dreamshaper_coreml_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    pipeline = _load_pipeline()

    variant = config["variant"]
    methods = pipeline.load_methods()
    inputs = pipeline.example_inputs(variant["methods"])
    programs = pipeline.exported_programs(methods, inputs)

    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]

    # CPU_AND_GPU, NOT the default ComputeUnit.ALL: with the ANE enabled the
    # `denoise` method fails to LOAD on device (`ANECCompile() FAILED`, backend
    # error 0x23) -- the SDXS UNet is too large for the ANE compiler. See NOTES.md.
    compile_specs = CoreMLBackend.generate_compile_specs(
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS17,
        compute_unit=ct.ComputeUnit.CPU_AND_GPU,
    )

    executorch_program = to_edge_transform_and_lower(
        programs,
        partitioner=[CoreMLPartitioner(compile_specs=compile_specs)],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(executorch_program.buffer)
    return str(out_path)
