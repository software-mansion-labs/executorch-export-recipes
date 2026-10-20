import importlib.util
import sys
from pathlib import Path

import torch



def _attention_rank3(self, hidden_states, **kwargs):
    """Avoid broadcast expansion copies for the fixed singleton-batch export."""
    batch, tokens, channels = hidden_states.shape
    assert batch == 1, "Depth export attention requires batch size one"
    heads = self.num_attention_heads
    head_dim = self.attention_head_size
    queries = self.query(hidden_states).reshape(tokens, heads, head_dim).permute(1, 0, 2)
    keys = self.key(hidden_states).reshape(tokens, heads, head_dim).permute(1, 0, 2)
    values = self.value(hidden_states).reshape(tokens, heads, head_dim).permute(1, 0, 2)

    # Scale the small query tensor, rather than the H x tokens x tokens scores.
    scores = torch.bmm(queries * self.scaling, keys.transpose(1, 2))
    weights = torch.softmax(scores, dim=-1)
    context = torch.bmm(weights, values).permute(1, 0, 2).reshape(1, tokens, channels)
    return context, weights.unsqueeze(0)


def _load_model():
    name = "depth_anything_v2_model_xnnpack"
    path = Path(__file__).resolve().parents[1] / "_model.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from transformers.models.dinov2.modeling_dinov2 import Dinov2SelfAttention

    model = module.build_model()
    for layer in model.modules():
        if isinstance(layer, Dinov2SelfAttention):
            layer.forward = _attention_rank3.__get__(layer)
    return model


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    model = _load_model()
    inputs = (torch.zeros(shape, dtype=torch.float32),)
    with torch.no_grad():
        if variant['quantized']:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer, get_symmetric_quantization_config,
            )
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

            # Static activation quantization badly distorts depth. Quantize only
            # transformer linears with dynamic activations; retain float convolutions.
            quantizer = XNNPACKQuantizer().set_module_type(
                torch.nn.Linear,
                get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True),
            )
            prepared = prepare_pt2e(torch.export.export(model, inputs, strict=False).module(), quantizer)
            # Activations are dynamic; one pass initializes the weight observers.
            prepared(*inputs)
            model = convert_pt2e(prepared)
        ep = torch.export.export(model, inputs, strict=False)
    program = to_edge_transform_and_lower(
        ep, partitioner=[XnnpackPartitioner(extended_ops=True)]
    ).to_executorch()

    path = Path(out_dir) / variant["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(program.buffer)
    return str(path)
