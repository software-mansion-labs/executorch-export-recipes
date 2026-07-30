import importlib.util
import sys
from pathlib import Path

import torch

from exporter.dynamic_dims import build_dynamic_dims_program

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

SEQ_LEN = 256
GROUP_SIZE = 32


def _load_pf_model():
    name = "pf_model_openai_xnnpack"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_pf_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import torch.nn as nn
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import (
        Int8DynamicActivationIntxWeightConfig,
        quantize_,
    )
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    m = _load_pf_model()
    model, tok = m.load_model(torch.float32)

    for layer in model.model.layers:
        src = layer.mlp.experts
        layer.mlp.experts = m.PrivacyFilterMoEDense(src, limit=src.limit, alpha=src.alpha)

    qcfg = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(GROUP_SIZE),
    )
    quantize_(model, qcfg, filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear))

    ids, mask = m.example_inputs(tok, SEQ_LEN)
    wrapper = m.ClassifierWrapper(model).eval()

    # Dynamic sequence length: the token-classification MoE runs every expert on
    # every token, so it is O(tokens) and padding a short input up to SEQ_LEN
    # wastes proportional compute. Export a symbolic sequence dim plus the
    # get_dynamic_dims_forward companion the runtime reads the active
    # [min, max, step] range from.
    seq = torch.export.Dim("tokens", min=2, max=SEQ_LEN)
    dynamic = {"input_ids": {1: seq}, "attention_mask": {1: seq}}
    ep = torch.export.export(wrapper, (ids, mask), dynamic_shapes=dynamic, strict=True)

    seq_bounds = [[1, 1, 1], [1, SEQ_LEN, 1]]
    dims = build_dynamic_dims_program([seq_bounds, seq_bounds])

    # Drop MaxDimConfig from the partitioner: the model's attention subtracts
    # combined_logits.max(dim=-1), and XNNPACK's amax visitor mis-serializes that
    # rank-4 max(dim=-1) into a max-pool (it never canonicalizes the negative
    # dim), silently corrupting the logits. There is only one such node per
    # layer, so leaving it on the portable CPU backend costs ~nothing. Verified:
    # 100% label agreement vs the PyTorch reference (delegating amax drops it to
    # ~25%).
    partitioner_configs = [
        c for c in ALL_PARTITIONER_CONFIGS if c.__name__ != "MaxDimConfig"
    ]
    et = to_edge_transform_and_lower(
        {"forward": ep, "get_dynamic_dims_forward": dims},
        partitioner=[XnnpackPartitioner(configs=partitioner_configs)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et.buffer)
    return str(out_path)
