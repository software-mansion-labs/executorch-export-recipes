"""Carry a quantized Echo checkpoint's own integer codes into torchao tensors.

The w8a16 / w8a8 / w4a16 checkpoints are QAT-trained: their codes and scales
are the model. torchao's symmetric scheme picks scale = amax / 127.5 (int8) or
amax / 7.5 (int4), and the w4a16 codes stop at +-7 with only 58% of groups
reaching it, so re-quantizing the dequantized weights would change the codes.
Instead each decoder linear is quantized with the matching torchao config and
its qdata / scale / zero_point are overwritten with the checkpoint's.

Activation handling is a backend choice, not a checkpoint property:
  dynamic=True  -> int8 per-token dynamic activations (XNNPACK and Vulkan
                   "8da" kernels). Matches w8a8's training; for w4a16 / w8a16
                   it adds activation rounding the model was not trained with.
  dynamic=False -> weight-only (fp activations), the checkpoint's "a16".
"""

import json
from pathlib import Path

import torch


def _codes(path: Path):
    from safetensors.torch import load_file

    config = json.loads((path / "config.json").read_text())
    q = config["quantization"]
    weights = load_file(str(path / "model.safetensors"))
    out = {}
    for key in [k for k in weights if k.endswith("_scale")]:
        name = key[: -len("_scale")]
        codes, scale = weights[name], weights[key].float()
        if q["bits"] == 4:  # two per byte, low nibble first, stored as code + 8
            codes = (torch.stack([codes & 0xF, codes >> 4], dim=-1).flatten(-2).to(torch.int16) - 8).to(torch.int8)
        out[name] = (codes, scale)
    return q, out


def quantize_from_checkpoint(model: torch.nn.Module, path, dynamic: bool) -> int:
    """Quantize `model`'s decoder linears in place with the checkpoint's codes.
    `model` must already hold the checkpoint's dequantized weights."""
    from torchao.quantization import MappingType, quantize_
    from torchao.quantization.granularity import PerAxis, PerGroup
    from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, IntxWeightOnlyConfig

    q, codes = _codes(Path(path))
    dtype = torch.int4 if q["bits"] == 4 else torch.int8
    gran = PerGroup(q["group_size"]) if q.get("group_size") else PerAxis(0)
    if dynamic:
        cfg = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=dtype, weight_granularity=gran, weight_mapping_type=MappingType.SYMMETRIC
        )
    else:
        cfg = IntxWeightOnlyConfig(weight_dtype=dtype, granularity=gran, mapping_type=MappingType.SYMMETRIC)

    targets = {name[: -len(".weight")] for name in codes}
    quantize_(model, cfg, filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and fqn in targets)

    done = 0
    for fqn, mod in model.named_modules():
        if fqn not in targets:
            continue
        c, s = codes[fqn + ".weight"]
        w = mod.weight
        assert w.qdata.shape == c.shape, (fqn, w.qdata.shape, c.shape)
        with torch.no_grad():
            w.qdata.copy_(c)
            w.scale.copy_(s.view_as(w.scale))
            w.zero_point.zero_()
        done += 1
    assert done == len(targets), (done, len(targets))
    return done
