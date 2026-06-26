import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent


def _load_craft():
    p = _HERE / "_craft.py"
    spec = importlib.util.spec_from_file_location("craft__craft", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["craft__craft"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> None:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from executorch.exir import to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    craft = _load_craft()
    CRAFT = craft.CRAFT
    copyStateDict = craft.copyStateDict

    weights_path = _HERE / "craft_mlt_25k.pth"
    model = CRAFT()
    model.load_state_dict(
        copyStateDict(torch.load(str(weights_path), map_location="cpu", weights_only=False))
    )
    model.eval()

    sizes = {"forward_1280": (1280, 1280), "forward_320": (320, 320)}

    fp32_eps = {}
    int8_eps = {}

    for method, (H, W) in sizes.items():
        sample_inputs = (torch.randn(1, 3, H, W),)

        ep_fp32 = torch.export.export(model, sample_inputs)
        fp32_eps[method] = ep_fp32

        qparams = get_symmetric_quantization_config(is_per_channel=False, is_qat=False)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(qparams)

        training_ep = ep_fp32.module()
        prepared_model = prepare_pt2e(training_ep, quantizer)

        with torch.no_grad():
            for _ in range(10):
                prepared_model(torch.randn(1, 3, H, W) * 0.5)

        quantized_model = convert_pt2e(prepared_model)
        int8_eps[method] = torch.export.export(quantized_model, sample_inputs)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for variant in config["variants"]:
        name = variant["file"]
        if "int8" in name:
            et_program = to_edge_transform_and_lower(
                int8_eps,
                partitioner=[XnnpackPartitioner()],
            ).to_executorch()
        else:
            et_program = to_edge_transform_and_lower(
                fp32_eps,
                partitioner=[XnnpackPartitioner()],
            ).to_executorch()
        (out / name).write_bytes(et_program.buffer)
