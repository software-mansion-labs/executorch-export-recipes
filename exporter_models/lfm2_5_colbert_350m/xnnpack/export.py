import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_lfm_retrieval_model():
    name = f"lfm_retrieval_model_{_MODEL_DIR.name}_xnnpack"
    spec = importlib.util.spec_from_file_location(
        name, _MODEL_DIR / "_lfm_retrieval_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.extension.llm.export.quantize import quantize_model_

    m = _load_lfm_retrieval_model()
    model, tok = m.load_colbert(torch.float32)

    ex_ids = torch.randint(2, 1000, (1, 16), dtype=torch.long)
    ex_mask = torch.ones_like(ex_ids)
    td = torch.export.Dim("t", min=2, max=512)
    dynamic_shapes = {"input_ids": {1: td}, "attention_mask": {1: td}}

    quantize_model_(model, qlinear_config="8da4w", qlinear_group_size=32)

    ep = torch.export.export(model, (ex_ids, ex_mask), dynamic_shapes=dynamic_shapes, strict=True)

    et_program = to_edge_transform_and_lower(
        ep,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et_program.buffer)
    return str(out_path)
