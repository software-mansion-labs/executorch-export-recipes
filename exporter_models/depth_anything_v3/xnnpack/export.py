import importlib.util
import sys
from pathlib import Path

import torch


def _load_model():
    name = "depth_anything_v3_model_xnnpack"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_model()


def _attention_rank3(self, x, pos=None, attn_mask=None):
    """Use explicit singleton-batch attention without SDPA mask bookkeeping."""
    batch, tokens, channels = x.shape
    assert batch == 1 and attn_mask is None
    heads = self.num_heads
    width = channels // heads
    qkv = self.qkv(x)
    q = qkv[..., :channels].reshape(tokens, heads, width).permute(1, 0, 2)
    k = qkv[..., channels : 2 * channels].reshape(tokens, heads, width).permute(1, 0, 2)
    v = qkv[..., 2 * channels :].reshape(tokens, heads, width).permute(1, 0, 2)
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None and pos is not None:
        q = self.rope(q.unsqueeze(0), pos).reshape(heads, tokens, width)
        k = self.rope(k.unsqueeze(0), pos).reshape(heads, tokens, width)
    scores = torch.bmm(q * self.scale, k.transpose(1, 2))
    context = torch.bmm(torch.softmax(scores, dim=-1), v)
    context = context.permute(1, 0, 2).reshape(1, tokens, channels)
    return self.proj_drop(self.proj(context))


def _static_rope(self, tokens, positions):
    # Positions are fixed by the static patch grid, never by image contents.
    # PT2E scalar-tensor normalization can promote the grid addition to float.
    positions = positions.to(torch.int64)
    feature_dim = tokens.size(-1) // 2
    cos, sin = self._compute_frequency_components(
        feature_dim, self._export_max_position, tokens.device, tokens.dtype
    )
    vertical, horizontal = tokens.chunk(2, dim=-1)
    return torch.cat(
        (
            self._apply_1d_rope(vertical, positions[..., 0], cos, sin),
            self._apply_1d_rope(horizontal, positions[..., 1], cos, sin),
        ),
        dim=-1,
    )


def _fold_constants(graph):
    """Fold image-independent positional tensors before PT2E scalar rewriting."""
    from torch.fx.node import map_arg

    values = {}
    for node in list(graph.graph.nodes):
        if node.op == "get_attr":
            value = graph
            for component in node.target.split("."):
                value = getattr(value, component)
            values[node] = value
        elif node.op == "call_function" and all(
            n in values for n in node.all_input_nodes
        ):
            args = map_arg(node.args, lambda n: values[n])
            kwargs = map_arg(node.kwargs, lambda n: values[n])
            value = node.target(*args, **kwargs)
            values[node] = value
            # Keep large spatial embeddings as broadcast expressions.
            if isinstance(value, torch.Tensor) and (
                value.numel() <= 100_000 or value.dtype in (torch.int8, torch.uint8)
            ):
                name = f"_export_constant_{len(values)}"
                graph.register_buffer(
                    name, value.detach().clone(memory_format=torch.contiguous_format)
                )
                with graph.graph.inserting_before(node):
                    replacement = graph.graph.get_attr(name)
                    replacement.meta = {
                        k: v
                        for k, v in node.meta.items()
                        if k not in ("val", "tensor_meta")
                    }
                node.replace_all_uses_with(replacement)
                values[replacement] = value
    graph.graph.eliminate_dead_code()
    graph.graph.lint()
    graph.recompile()


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    model = _load_model()
    from depth_anything_3.model.dinov2.layers.attention import Attention
    from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D

    for layer in model.modules():
        if isinstance(layer, Attention):
            layer.forward = _attention_rank3.__get__(layer)
        if isinstance(layer, RotaryPositionEmbedding2D):
            layer._export_max_position = max(shape[-2:]) // 14 + 1
            layer.forward = _static_rope.__get__(layer)
    torch.set_num_threads(4)
    inputs = (torch.zeros(shape, dtype=torch.float32),)
    with torch.no_grad():
        model(*inputs)
        # Functionalize positional arithmetic before folding it into buffers.
        graph = (
            torch.export.export(model, inputs, strict=False)
            .run_decompositions({})
            .module()
        )
        _fold_constants(graph)
        model = graph
        if variant["quantized"]:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )
            from torchao.quantization.pt2e.quantize_pt2e import (
                convert_pt2e,
                prepare_pt2e,
            )

            quantizer = XNNPACKQuantizer().set_module_type(
                torch.nn.Linear,
                get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True),
            )
            print("DA3: preparing INT8 observers", flush=True)
            prepared = prepare_pt2e(graph, quantizer)
            print("DA3: initializing INT8 weight observers", flush=True)
            prepared(*inputs)
            print("DA3: converting INT8 graph", flush=True)
            model = convert_pt2e(prepared, fold_quantize=False)
            _fold_constants(model)
            print("DA3: INT8 conversion complete", flush=True)
        ep = torch.export.export(model, inputs, strict=False)
        assert not ep.graph_signature.buffers_to_mutate
    program = to_edge_transform_and_lower(
        ep, partitioner=[XnnpackPartitioner(extended_ops=True)]
    ).to_executorch()
    path = Path(out_dir) / variant["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(program.buffer)
    return str(path)
