import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent


def _load_lfm_retrieval_model():
    name = f"lfm_retrieval_model_{_HERE.name}"
    spec = importlib.util.spec_from_file_location(
        name, _HERE / "_lfm_retrieval_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    m = _load_lfm_retrieval_model()

    if config["backend"] == "mlx":
        from executorch.backends.mlx.partitioner import MLXPartitioner
        from executorch.extension.llm.export.quantize import quantize_model_

        model, tok = m.load_embedding(torch.bfloat16)
        partitioner = MLXPartitioner()

        def _quantize(model):
            quantize_model_(model, qlinear_config="4w", qlinear_group_size=64)
    else:
        import torch.nn as nn
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        model, tok = m.load_embedding(torch.float32)
        partitioner = XnnpackPartitioner()

        def _quantize(model):
            qcfg = Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(32),
            )
            quantize_(model, qcfg, filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear) and ".conv." not in fqn)

    ids, mask = m.encode_inputs(tok, ["hello world"], "document: ", 32)
    token_dim = torch.export.Dim("token_dim", min=2, max=512)
    dynamic_shapes = {"input_ids": {1: token_dim}, "attention_mask": {1: token_dim}}

    _quantize(model)

    ep = torch.export.export(model, (ids, mask), dynamic_shapes=dynamic_shapes, strict=True)

    et_program = to_edge_transform_and_lower(
        ep,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et_program.buffer)
    return str(out_path)
