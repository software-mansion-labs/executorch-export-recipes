import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_pipeline():
    path = _MODEL_DIR / "_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sdxs_512_dreamshaper_xnnpack_pipeline", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sdxs_512_dreamshaper_xnnpack_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    pipeline = _load_pipeline()

    variant = config["variant"]
    methods = pipeline.load_methods()
    inputs = pipeline.example_inputs(variant["methods"])
    programs = pipeline.exported_programs(methods, inputs)

    # fp32 only. SDXS is extremely quantization-sensitive (see NOTES.md): the
    # ~14.6x scheduler amplification of the single distilled step magnifies any
    # weight error, and quantizing the convolutions destroys the image.
    executorch_program = to_edge_transform_and_lower(
        programs,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(executorch_program.buffer)
    return str(out_path)
