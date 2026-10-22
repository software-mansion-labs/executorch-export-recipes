import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_lfm_retrieval_model():
    name = f"lfm_retrieval_model_{_MODEL_DIR.name}_coreml"
    spec = importlib.util.spec_from_file_location(
        name, _MODEL_DIR / "_lfm_retrieval_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]

    m = _load_lfm_retrieval_model()
    # float32 rather than the bfloat16 the MLX path uses -- Core ML has no bf16
    # input dtype. No quantize_model_ either: that packs weights for the MLX
    # runtime, so the Core ML variant ships unquantized at the chosen precision.
    model, _tok = m.load_colbert(torch.float32)
    model.eval()

    ex_ids = torch.randint(2, 1000, (1, 16), dtype=torch.long)
    ex_mask = torch.ones_like(ex_ids)
    td = torch.export.Dim("t", min=2, max=max_tokens)
    dynamic_shapes = {"input_ids": {1: td}, "attention_mask": {1: td}}

    ep = torch.export.export(
        model, (ex_ids, ex_mask), dynamic_shapes=dynamic_shapes, strict=True
    )

    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )
    program = to_edge_transform_and_lower(
        ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
