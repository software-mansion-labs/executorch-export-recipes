import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_pipeline():
    path = _MODEL_DIR / "_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sdxs_512_dreamshaper_vulkan_pipeline", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sdxs_512_dreamshaper_vulkan_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    pipeline = _load_pipeline()

    variant = config["variant"]
    methods = pipeline.load_methods()
    inputs = pipeline.example_inputs(variant["methods"])
    programs = pipeline.exported_programs(methods, inputs)

    # fp16, and NOTES.md's quantization warning does not apply to it. The ~14.6x
    # scheduler amplification does sink int8, but measured against the shipped
    # fp32 .pte, fp16 `denoise` is corr 0.999985 and `decode` 0.999997, and the
    # end-to-end image is 3.6/255 mean abs error (cos 0.9989). fp16 is also the
    # only arm that wins: Vulkan fp32 `denoise` is 0.934x, a loss, while fp16 is
    # 1.34x end to end on Adreno 840.
    #
    # fp16 REQUIRES a runtime carrying pytorch/executorch#22827. Without it the
    # CLIP text embedding (vocab 49408) is served by the broken height-packed
    # embedding_texture path and `encode` returns cos 0.05 garbage.
    executorch_program = to_edge_transform_and_lower(
        programs,
        partitioner=[VulkanPartitioner(compile_options={"force_fp16": True})],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(executorch_program.buffer)
    return str(out_path)
