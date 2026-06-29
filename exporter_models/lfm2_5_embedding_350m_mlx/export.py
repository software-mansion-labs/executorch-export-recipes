import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent


def _load_lfm_retrieval_model():
    spec = importlib.util.spec_from_file_location(
        "lfm_retrieval_model", _HERE / "_lfm_retrieval_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lfm_retrieval_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.extension.llm.export.quantize import quantize_model_
    from torch.export import Dim

    m = _load_lfm_retrieval_model()
    model, tok = m.load_embedding(torch.float32)

    ids, mask = m.encode_inputs(tok, ["sample query"], prefix="", seq_len=64)

    seq_dim = Dim("seq", min=2, max=512)
    dynamic_shapes = ({1: seq_dim}, {1: seq_dim})

    quantize_model_(model, qlinear_config="4w", qlinear_group_size=64)

    ep = torch.export.export(model, (ids, mask), dynamic_shapes=dynamic_shapes, strict=True)

    et_program = to_edge_transform_and_lower(
        ep,
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et_program.buffer)
    return str(out_path)
