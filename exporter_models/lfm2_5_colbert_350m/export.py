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
    from executorch.extension.llm.export.quantize import quantize_model_

    m = _load_lfm_retrieval_model()

    if config["backend"] == "mlx":
        from executorch.backends.mlx.partitioner import MLXPartitioner

        model, tok = m.load_colbert(torch.bfloat16)
        partitioner = MLXPartitioner()
        qlinear_config, qlinear_group_size = "4w", 64
    else:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        model, tok = m.load_colbert(torch.float32)
        partitioner = XnnpackPartitioner()
        qlinear_config, qlinear_group_size = "8da4w", 32

    ex_ids = torch.randint(2, 1000, (1, 16), dtype=torch.long)
    ex_mask = torch.ones_like(ex_ids)
    td = torch.export.Dim("t", min=2, max=512)
    dynamic_shapes = {"input_ids": {1: td}, "attention_mask": {1: td}}

    quantize_model_(model, qlinear_config=qlinear_config, qlinear_group_size=qlinear_group_size)

    ep = torch.export.export(model, (ex_ids, ex_mask), dynamic_shapes=dynamic_shapes, strict=True)

    et_program = to_edge_transform_and_lower(
        ep,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et_program.buffer)
    return str(out_path)
