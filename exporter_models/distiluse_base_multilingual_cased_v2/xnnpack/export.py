import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import Dim, export as torch_export

from exporter.backends import lower
from exporter.dynamic_dims import build_dynamic_dims_program

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_MAX_TOKENS = 126  # model max_seq_length 128 - 2 for CLS/SEP


def _load_model():
    name = "distiluse_xnnpack_model"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_distiluse_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import torch.nn as nn
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_

    variant = config["variant"]
    module = _load_model().load_embedder()

    inputs = (
        torch.zeros(1, 8, dtype=torch.int64),
        torch.ones(1, 8, dtype=torch.int64),
    )
    dim = Dim("tokens", min=1, max=_MAX_TOKENS)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    if variant["quantized"]:
        qcfg = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4, weight_granularity=PerGroup(32)
        )
        quantize_(module, qcfg, filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear))

    ep = torch_export(module, inputs, dynamic_shapes=dynamic)

    # Advertise the dynamic sequence dimension to the runtime. Both inputs are
    # [1, tokens]: batch static, tokens dynamic in [1, _MAX_TOKENS].
    seq_bounds = [[1, 1, 1], [1, _MAX_TOKENS, 1]]
    dims = build_dynamic_dims_program([seq_bounds, seq_bounds])
    program = lower({"forward": ep, "get_dynamic_dims_forward": dims}, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
