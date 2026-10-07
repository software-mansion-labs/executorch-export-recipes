import importlib.util
import sys
from pathlib import Path

import torch

from exporter.schema import load_and_validate_schema

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
    import torch.nn as nn
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_

    m = _load_lfm_retrieval_model()
    model, tok = m.load_colbert(torch.float32)
    # transformers >= 5.x passes a `seq_idx` kwarg the model's remote short-conv
    # override rejects; re-patch to tolerate it (numerics unchanged).
    m.patch_shortconv_static_length(model)

    ex_ids = torch.randint(2, 1000, (1, 16), dtype=torch.long)
    ex_mask = torch.ones_like(ex_ids)
    td = torch.export.Dim("t", min=2, max=512)
    dynamic_shapes = {"input_ids": {1: td}, "attention_mask": {1: td}}

    # Leave the short-conv linears in fp32, as the lfm2_5_embedding_350m sibling
    # does. Quantizing them too is 144 MB smaller but measurably less faithful
    # (see NOTES.md).
    qcfg = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4, weight_granularity=PerGroup(32))
    quantize_(model, qcfg, filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear) and ".conv." not in fqn)

    # 8da4w covers the Linears only, which leaves the fp32 word-embedding table
    # ([64402, 1024], 264 MB) in the file. Quantize it weight-only to int8 per row
    # (embedding_byte, a portable kernel linked into the RNE runtime): 570 -> 373 MB
    # at no measurable cost, where int4 g32 saves 25 MB more for a little fidelity.
    from executorch.examples.models.llama.source_transformation.quantize import (
        EmbeddingQuantHandler,
    )

    model = EmbeddingQuantHandler(model, bitwidth=8, precision=torch.float32).quantized_model()

    ep = torch.export.export(model, (ex_ids, ex_mask), dynamic_shapes=dynamic_shapes, strict=True)

    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")

    et_program = to_edge_transform_and_lower(
        {"forward": ep},
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        constant_methods={"get_model_schema": schema_json},
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et_program.buffer)
    return str(out_path)
