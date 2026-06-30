import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

SEQ_LEN = 256
GROUP_SIZE = 32


def _load_pf_model():
    name = "pf_model_nemotron_xnnpack"
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

    ep = torch.export.export(wrapper, (ids, mask), strict=True)
    et = to_edge_transform_and_lower(
        ep,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et.buffer)
    return str(out_path)
