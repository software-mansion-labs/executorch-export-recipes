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


def _quantize_dynamic(program, inputs):
    import torch
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )
    with torch.no_grad():
        prepared = prepare_pt2e(program.module(), quantizer)
        prepared(*inputs)  # dynamic: only the weight observers need a pass
        return torch.export.export(convert_pt2e(prepared), inputs, strict=True)


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

    partitioners = [XnnpackPartitioner()]
    if variant.get("quantize"):
        # Static int8 destroys the image (see NOTES.md); dynamic int8 activations
        # with int8 per-channel weights on the convs and Linears of the UNet and
        # the TAESD decoder do not. The text encoder stays fp32: its error would
        # be amplified ~14.6x by the single denoising step, for 4% of the time.
        assert variant["quantize"] == "8da8w-pc", variant["quantize"]
        for name in ("denoise", "decode"):
            programs[name] = _quantize_dynamic(programs[name], inputs[name])
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
        )
        # One partitioner over mixed dynamic-quant and fp32 nodes forms cyclic partitions.
        partitioners = [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]

    executorch_program = to_edge_transform_and_lower(
        programs,
        partitioner=partitioners,
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(executorch_program.buffer)
    return str(out_path)
