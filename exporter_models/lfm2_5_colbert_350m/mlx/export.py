import importlib.util
import sys
from pathlib import Path

import torch

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_lfm_retrieval_model():
    name = f"lfm_retrieval_model_{_MODEL_DIR.name}_mlx"
    spec = importlib.util.spec_from_file_location(
        name, _MODEL_DIR / "_lfm_retrieval_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.extension.llm.export.quantize import quantize_model_

    m = _load_lfm_retrieval_model()
    model, tok = m.load_colbert(torch.bfloat16)

    ex_ids = torch.randint(2, 1000, (1, 16), dtype=torch.long)
    ex_mask = torch.ones_like(ex_ids)
    td = torch.export.Dim("t", min=2, max=512)
    dynamic_shapes = {"input_ids": {1: td}, "attention_mask": {1: td}}

    # Quantize the embedding table too, matching the sibling
    # lfm2_5_embedding_350m exporter. Linear-only quantization left the token
    # embeddings in bf16 and the artifact at 284.9 MB; adding 4w/64 embeddings
    # takes it to 188.0 MB (-34%) with no measurable quality cost (per-token
    # cosine 0.99327 vs 0.99363, MaxSim retrieval identical) and +0.4 ms on an
    # iPhone 16 at 128 tokens (61.3 vs 60.9 ms median).
    quantize_model_(
        model,
        qlinear_config="4w",
        qlinear_group_size=64,
        qembedding_config="4w",
        qembedding_group_size=64,
    )

    ep = torch.export.export(model, (ex_ids, ex_mask), dynamic_shapes=dynamic_shapes, strict=True)

    # transform_passes AND _skip_dim_order are both required, and only together:
    # without the pair the lowered program fails to load in the ExecuTorch runtime
    # with `Failed to load method forward, error: 0x23`. Verified as a full truth
    # table -- passes alone, _skip_dim_order alone, and either combined with the
    # dict call form all still fail; only both together load. This matches the
    # sibling lfm2_5_embedding_350m exporter, which has carried both all along.
    # Ship the companion schema method. Without it the runtime falls back to
    # ExecuTorch MethodMeta, where every dim is `constant` and no runtime
    # constraint is expressed -- so the seq range {2..512} and the equality
    # tying input_ids/attention_mask/output seq dims together are both lost, and
    # spec validation rejects the model. The xnnpack sibling already does this.
    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")

    et_program = to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
        constant_methods={"get_model_schema": schema_json},
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et_program.buffer)
    return str(out_path)
