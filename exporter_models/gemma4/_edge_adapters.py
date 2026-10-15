"""Adapters that fit the `_gemma4_edge` text recipes to the RNE text runner.

The vendored `_gemma4_edge` tree must stay byte-identical to gemma4_omni's, so
these patch names inside its `pipeline` module instead of editing it. Used by
the XNNPACK and Vulkan text exporters.
"""

import importlib
import importlib.util
import inspect
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent

# Module-level holder for top-level graph input start_pos during export trace.
# Attention ops read start_pos_arg = _GRAPH_START_POS_HOLDER[0][0].item(), which traces to
# aten.select(input_pos, 0, 0) -> aten.item(). This guarantees Vulkan has a host
# staging buffer for start_pos_arg without any intermediate GPU compute tensors.
_GRAPH_START_POS_HOLDER = [None]


def adapt(pipeline) -> None:
    """Applies both adapters below to an imported `_gemma4_edge` pipeline."""
    scalar_input_pos(pipeline)
    lean_loader(pipeline)


def lean_loader(pipeline):
    """Routes the pipeline's checkpoint load through `_lean_load.lean_load`.

    The edge port builds the model in fp32 and copies a full state dict into it
    before casting to `BASE_DTYPE`; that does not fit in 16 GB. Only the loader
    is wrapped, so quantisation and export run exactly as before.
    """
    spec = importlib.util.spec_from_file_location("gemma4_lean_load", _HERE / "_lean_load.py")
    lean = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lean)
    orig = pipeline.load_truncated_from_safetensors

    def load(*args, **kwargs):
        with lean.lean_load():
            return orig(*args, **kwargs)

    pipeline.load_truncated_from_safetensors = load


def scalar_input_pos(pipeline):
    """Adapts the text `forward` to the RNE text runner contract, in place.

    The edge port traces `forward(input_ids[1, S], input_pos[S]) -> [1, S|1,
    vocab]`; the runner calls `forward(tokens[1, S], pos[1]) -> [1, vocab]`. The
    vendored `_gemma4_edge` tree must stay byte-identical to gemma4_omni's, so
    instead of editing it this swaps two names inside `pipeline`: the forward
    wrapper (expand the start position, keep the last logits) and the trace call
    (a static [1] position instead of the seq-ranged one).
    """
    base = pipeline.SingleMethodForwardExport

    class ScalarPosForward(base):
        def forward(self, input_ids, input_pos):
            # input_pos[:1] is the start position whether the caller passes the
            # runner's [1] or the eager probe's [S] positions. The 0..S-1
            # offsets come from a cumsum over the tokens rather than
            # `arange(S)`: arange needs the symbolic seq length, and a Vulkan
            # partition whose only output is that SymInt fails memory planning
            # ("graph_output_allocated not set").
            start_pos = input_pos[:1]
            _GRAPH_START_POS_HOLDER[0] = start_pos
            offsets = torch.cumsum(torch.ones_like(input_ids[0]), dim=0) - 1
            positions = start_pos + offsets

            logits = super().forward(input_ids, positions)
            if logits.dim() == 3:
                logits = logits.squeeze(1) if logits.shape[1] == 1 else logits[:, -1, :]
            return logits.to(torch.float32)

    # Patch sdpa and rope transforms in _gemma4_edge so start_pos_arg reads from _GRAPH_START_POS_HOLDER
    _patch_vulkan_transforms()

    orig_export = pipeline.torch_export

    def export_with_scalar_pos(module, args, *rest, dynamic_shapes=None, **kwargs):
        if isinstance(module, ScalarPosForward):
            input_ids, input_pos = args
            args = (input_ids, input_pos[:1].clone())
            dynamic_shapes = dict(dynamic_shapes, input_pos={})
        return orig_export(module, args, *rest, dynamic_shapes=dynamic_shapes, **kwargs)

    pipeline.SingleMethodForwardExport = ScalarPosForward
    pipeline.torch_export = export_with_scalar_pos


def _patch_vulkan_transforms(edge_pkg: str | None = None):
    """Patches sdpa.py and rope.py so start_pos_arg reads top-level graph input start_pos
    and sliding masks use F.embedding instead of broken aten.index.Tensor.
    """
    pkgs = [edge_pkg] if edge_pkg else ["gemma4_edge_text", "gemma4_edge_omni"]
    for pkg in pkgs:
        try:
            sdpa_mod = importlib.import_module(f"{pkg}.model.transforms.sdpa")
            rope_mod = importlib.import_module(f"{pkg}.model.transforms.rope")
        except ImportError:
            continue
        _apply_vulkan_transforms_patch(sdpa_mod, rope_mod)


def _apply_vulkan_transforms_patch(sdpa_mod, rope_mod):
    def _patched_build_mask4d_for_sliding(attn_module, input_pos, q_dtype):
        use_bool_mask = getattr(type(attn_module), "use_bool_mask", False)
        # Use F.embedding instead of attn_module.mask[input_pos] (aten.index.Tensor).
        # F.embedding lowers to aten.embedding.default, which Vulkan supports natively.
        # aten.index.Tensor in Vulkan only supports 1D textures and produces corrupted
        # coordinates for 2D attention masks.
        mask_row = torch.nn.functional.embedding(input_pos, attn_module.mask)
        if use_bool_mask:
            mask_row = torch.zeros_like(
                mask_row, dtype=q_dtype,
            ).masked_fill(mask_row, float("-inf"))
        else:
            mask_row = mask_row.to(q_dtype)
        return mask_row.unsqueeze(0).unsqueeze(0).contiguous()

    sdpa_mod._build_mask4d_for_sliding = _patched_build_mask4d_for_sliding
    rope_mod._build_mask4d_for_sliding = _patched_build_mask4d_for_sliding

    replacement = (
        'start_pos_arg = (_GRAPH_START_POS_HOLDER[0][0].item() '
        'if _GRAPH_START_POS_HOLDER[0] is not None else input_pos[0].item())'
    )

    # Patch sdpa._make_standalone_or_source_forward
    src = inspect.getsource(sdpa_mod._make_standalone_or_source_forward)
    src = src.replace('start_pos_arg = input_pos[0].item()', replacement)
    scope = sdpa_mod.__dict__.copy()
    scope['_GRAPH_START_POS_HOLDER'] = _GRAPH_START_POS_HOLDER
    scope['_build_mask4d_for_sliding'] = _patched_build_mask4d_for_sliding
    exec(src, scope)
    sdpa_mod._make_standalone_or_source_forward = scope['_make_standalone_or_source_forward']

    # Patch sdpa._make_consumer_forward
    src = inspect.getsource(sdpa_mod._make_consumer_forward)
    src = src.replace('start_pos_arg = input_pos[0].item()', replacement)
    scope = sdpa_mod.__dict__.copy()
    scope['_GRAPH_START_POS_HOLDER'] = _GRAPH_START_POS_HOLDER
    scope['_build_mask4d_for_sliding'] = _patched_build_mask4d_for_sliding
    exec(src, scope)
    sdpa_mod._make_consumer_forward = scope['_make_consumer_forward']

    # Patch rope._make_paired_rope_standalone_forward
    src = inspect.getsource(rope_mod._make_paired_rope_standalone_forward)
    src = src.replace('start_pos_arg = input_pos[0].item()', replacement)
    scope = rope_mod.__dict__.copy()
    scope['_GRAPH_START_POS_HOLDER'] = _GRAPH_START_POS_HOLDER
    scope['_build_mask4d_for_sliding'] = _patched_build_mask4d_for_sliding
    exec(src, scope)
    rope_mod._make_paired_rope_standalone_forward = scope['_make_paired_rope_standalone_forward']




def vulkan_post_fusion_dtype_promotion():
    """Normalize fused dtypes and use the dynamically resized conversion op.

    Quantized embedding fusion changes fp16 output to fp32. Vulkan binary
    and RMSNorm kernels require matching tensor dtypes. The ordinary
    _to_copy implementation uses a texture-only blit without a resize
    callback, whereas _to_dim_order_copy supports buffer casts and resizing.
    """
    import executorch.backends.vulkan.vulkan_preprocess as preprocess
    from executorch.backends.transforms.utils import get_param_tensor
    from executorch.backends.vulkan import utils
    from executorch.backends.vulkan._passes.insert_dtype_promotion import (
        InsertDtypePromotionPass,
    )
    from executorch.exir.dialects._ops import ops
    from executorch.exir.pass_base import PassResult

    original = preprocess.FusePatternsPass
    if getattr(original, "_gemma_post_fusion_promotion", False):
        return

    class FuseAndPromote(original):
        _gemma_post_fusion_promotion = True

        def call(self, graph_module):
            fused = super().call(graph_module)
            gm = fused.graph_module
            for node in gm.graph.nodes:
                if node.op != "call_function" or node.target != ops.edge.aten.rms_norm.default:
                    continue
                activation, weight = node.args[0], node.args[2]
                if not isinstance(weight, torch.fx.Node):
                    continue
                dtype = activation.meta["val"].dtype
                if weight.meta["val"].dtype == dtype:
                    continue
                tensor = get_param_tensor(self._exported_program, weight)
                if tensor is None:
                    raise ValueError("Vulkan RMSNorm weight must be a constant")
                utils.update_program_state_dict(
                    self._exported_program, weight.name, tensor.to(dtype)
                )
                weight.meta["val"] = weight.meta["val"].to(dtype)

            gm = InsertDtypePromotionPass()(gm).graph_module
            for node in gm.graph.nodes:
                if node.op != "call_function" or node.target != ops.edge.aten._to_copy.default:
                    continue
                kwargs = dict(node.kwargs)
                memory_format = kwargs.pop("memory_format", None)
                if memory_format not in (None, torch.contiguous_format, torch.preserve_format):
                    raise ValueError("Unexpected memory format in Gemma Vulkan cast")
                kwargs["dim_order"] = list(range(node.args[0].meta["val"].dim()))
                node.target = ops.edge.dim_order_ops._to_dim_order_copy.default
                node.kwargs = kwargs
            gm.recompile()
            return PassResult(gm, True)

    preprocess.FusePatternsPass = FuseAndPromote
