"""Shared Vulkan export helpers."""

from contextlib import contextmanager


@contextmanager
def embedding_off_texture():
    """Keeps `aten.embedding` on buffer storage while `force_fp16` is set.

    Under `force_fp16`, `TagMemoryMetaPass.constrain_op_arg_repset` constrains
    every argument to `ANY_TEXTURE` unconditionally. That moves an embedding's
    output off the correct `embedding_buffer` kernel and onto
    `embedding_texture3d_*`, which returns silently wrong values once the
    vocabulary exceeds the device's `maxImageDimension2D` (16384). The output is
    still a well-formed unit vector, so this corrupts retrieval quality without
    raising anything: cosine against the CPU reference drops to 0.51 for the
    CLIP text encoder (vocab 49408) and 0.67 for all-MiniLM-L6-v2 (30522).

    Exempting embedding from that one constraint keeps it on the buffer kernel
    while the rest of the graph still lowers to fp16 and texture. Cosine returns
    to >= 0.99995 on both Adreno 840 and Mali-G76.

    Export-side only: the emitted .pte is ordinary and needs no runtime change.

    Remove this once pytorch/executorch#22333 is fixed and the pin moves to a
    release carrying the fix.
    """
    from executorch.backends.vulkan._passes import tag_memory_meta_pass as pass_mod

    original = pass_mod.TagMemoryMetaPass.constrain_op_arg_repset

    def _is_embedding(node) -> bool:
        return (
            node is not None
            and getattr(node, "op", None) == "call_function"
            and "embedding" in str(getattr(node, "target", ""))
        )

    def constrain_op_arg_repset(self, arg_i, op_repsets):
        arg = op_repsets.op_node.args[arg_i]
        if isinstance(arg, list) and arg:
            arg = arg[0]
        # Both directions matter: the embedding node's own args, and the
        # consumer whose arg *is* the embedding's output.
        exempt = _is_embedding(op_repsets.op_node) or _is_embedding(arg)
        if not exempt:
            return original(self, arg_i, op_repsets)
        saved, self.force_fp16 = self.force_fp16, False
        try:
            return original(self, arg_i, op_repsets)
        finally:
            self.force_fp16 = saved

    pass_mod.TagMemoryMetaPass.constrain_op_arg_repset = constrain_op_arg_repset
    try:
        yield
    finally:
        pass_mod.TagMemoryMetaPass.constrain_op_arg_repset = original


def _bmm_attention(query, key, value, heads, head_dim, tokens, scale, bias):
    """Shared rank-3 attention body: [H, T, D] batched matmuls, no 4-D tensors."""
    import torch

    def split(x):
        return x.view(tokens, heads, head_dim).permute(1, 0, 2)

    scores = torch.bmm(split(query) * scale, split(key).transpose(1, 2))
    if bias is not None:
        scores = scores + bias
    weights = torch.softmax(scores, dim=-1)
    return (
        torch.bmm(weights, split(value))
        .permute(1, 0, 2)
        .reshape(1, tokens, heads * head_dim)
    )


def use_rank3_attention(model) -> int:
    """Rewrites self-attention to rank-3, folding the singleton batch into heads.

    The stock HF attention modules reshape every projection to 4-D [B, H, T, D].
    On Vulkan each of those reshapes becomes its own materialised view dispatch,
    and with B == 1 the batch axis carries no information. Running the same
    math rank-3 as [H, T, D] batched matmuls is bit-identical on CPU and
    measured 14% faster on Adreno 840 and 9% on Mali-G76 for all-MiniLM-L6-v2
    at 254 tokens.

    Handles the BERT, MPNet and DistilBERT attention modules. Returns the number
    of modules patched, and raises if that is zero: a transformers change that
    renames these classes would otherwise silently cost the speedup.
    """
    import types

    import torch

    def bert_forward(self, hidden_states, attention_mask=None, **kwargs):
        _, tokens, _ = hidden_states.shape
        out = _bmm_attention(
            self.query(hidden_states), self.key(hidden_states), self.value(hidden_states),
            self.num_attention_heads, self.attention_head_size, tokens, self.scaling,
            None if attention_mask is None else attention_mask[0],
        )
        return out, None

    def mpnet_forward(self, hidden_states, attention_mask=None, position_bias=None, **kwargs):
        _, tokens, _ = hidden_states.shape
        bias = None
        if position_bias is not None:
            bias = position_bias[0]
        if attention_mask is not None:
            bias = attention_mask[0] if bias is None else bias + attention_mask[0]
        out = _bmm_attention(
            self.q(hidden_states), self.k(hidden_states), self.v(hidden_states),
            self.num_attention_heads, self.attention_head_size, tokens,
            self.attention_head_size**-0.5, bias,
        )
        return (self.o(out),)

    def distilbert_forward(self, hidden_states, attention_mask=None, **kwargs):
        # DistilBERT folds the output projection into the attention module
        # itself, so out_lin is applied here rather than by a separate
        # BertSelfOutput. The second return value is attn_weights, which the
        # block discards.
        _, tokens, _ = hidden_states.shape
        out = _bmm_attention(
            self.q_lin(hidden_states), self.k_lin(hidden_states), self.v_lin(hidden_states),
            self.n_heads, self.attention_head_size, tokens, self.scaling,
            None if attention_mask is None else attention_mask[0],
        )
        return self.out_lin(out), None

    patched = 0
    for module in model.modules():
        name = type(module).__name__
        if name == "BertSelfAttention":
            module.forward = types.MethodType(bert_forward, module)
            patched += 1
        elif name == "MPNetSelfAttention":
            module.forward = types.MethodType(mpnet_forward, module)
            patched += 1
        elif name == "DistilBertSelfAttention":
            module.forward = types.MethodType(distilbert_forward, module)
            patched += 1
    if patched == 0:
        raise RuntimeError(
            "use_rank3_attention patched nothing; the attention module names "
            "changed and the rewrite would silently be a no-op"
        )
    return patched


@contextmanager
def _scalars_keep_dtype():
    """Stops the Vulkan quantizer lifting integer scalars as float32.

    `_convert_scalars_to_attrs` rewrites every scalar operand of `add`/`mul`
    with `torch.tensor(float(arg))`, so an int64 position-id add becomes a
    float32 buffer and the `embedding` that consumes it raises

        Expected tensor for argument #1 'indices' to have one of the following
        scalar types: Long, Int; but got torch.FloatTensor

    Keeping the node's own output dtype is the fix. Remove this once
    pytorch/executorch#22372 lands and the pin moves past it.
    """
    import torch
    from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix
    from executorch.backends.vulkan.quantizer import vulkan_quantizer as q
    from executorch.backends.vulkan.quantizer import vulkan_quantizer_utils as u

    def _convert_scalars_to_attrs(model):
        for n in model.graph.nodes:
            if n.op != "call_function" or n.target not in [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.mul.Tensor,
            ]:
                continue
            new_args = []
            for arg in n.args:
                if isinstance(arg, torch.fx.Node):
                    new_args.append(arg)
                    continue
                name = get_new_attr_name_with_prefix("_tensor_constant_")(model)
                scalar_tensor = torch.tensor(arg, dtype=n.meta["val"].dtype)
                model.register_buffer(name, scalar_tensor)
                with model.graph.inserting_before(n):
                    node = model.graph.create_node("get_attr", name, (), {})
                    node.meta["val"] = n.meta["val"].fake_mode.from_tensor(
                        scalar_tensor, static_shapes=True
                    )
                    new_args.append(node)
            n.args = tuple(new_args)
        model.recompile()
        return model

    originals = (u._convert_scalars_to_attrs, q._convert_scalars_to_attrs)
    u._convert_scalars_to_attrs = _convert_scalars_to_attrs
    q._convert_scalars_to_attrs = _convert_scalars_to_attrs
    try:
        yield
    finally:
        u._convert_scalars_to_attrs, q._convert_scalars_to_attrs = originals


def quantize_weight_only_int8(module, inputs, dynamic_shapes, filter_fn=None):
    """Quantizes every `nn.Linear` weight to int8, per output channel.

    Weight only, deliberately. `VulkanQuantizer`'s dynamic mode scales
    activations per tensor (`linear_q8ta_q8csw` takes one input scale), which a
    transformer encoder does not survive: mean cosine against the fp32 model
    drops to 0.91 on all-mpnet-base-v2, against 0.9993 for weight only. See
    pytorch/executorch#22431.

    The `.pte` this produces needs pytorch/executorch#22430 in the runtime;
    before that fix `aten._weight_int8pack_mm` dispatched to `linear_qcs8w_*`
    shaders that no longer exist and every execute aborted.

    `filter_fn` takes an fx `Node` and returns whether to quantize it, for
    models where some linears must stay in float (LFM2's short-conv projections
    scramble the embedding when quantized). Default is every linear.

    Returns a graph module that `torch.export` re-exports with the same dynamic
    shapes it was given.
    """
    import torch
    from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
        get_symmetric_quantization_config,
        VulkanQuantizer,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = VulkanQuantizer().set_global(
        get_symmetric_quantization_config(is_dynamic=False, weight_bits=8)
    )
    if filter_fn is not None:
        quantizer.set_filter_function(filter_fn)
    with _scalars_keep_dtype():
        # Export with the dynamic dims already in place: a static export here
        # bakes the bound into the converted graph and the re-export then
        # rejects the Dim.
        prepared = prepare_pt2e(
            torch.export.export(module, inputs, dynamic_shapes=dynamic_shapes).module(),
            quantizer,
        )
        # Only the per-channel weight observers need feeding; activations are
        # untouched. Two lengths keep the token dim from re-specializing.
        max_tokens = inputs[0].shape[-1]
        torch.manual_seed(0)
        for length in (max_tokens, max(1, max_tokens // 3)):
            prepared(
                torch.randint(4, 30000, (1, length), dtype=torch.int64),
                torch.ones(1, length, dtype=torch.int64),
            )
        return convert_pt2e(prepared)


def _plain_softmax(x, dim, dtype=None):
    """``aten._safe_softmax`` without its masked-row guard."""
    import torch

    return torch.softmax(x, dim=dim, dtype=dtype)


def attention_decompositions_without_masked_row_guard() -> dict:
    """Decomposition table that drops the dead masked-row guard from attention.

    Since PyTorch 2.5 the SDPA decomposition emits ``aten._safe_softmax``, which
    follows the softmax with a guard that zeroes any row whose scores were all
    ``-inf``::

        masked_rows = (scores == -inf).all(dim, keepdim=True)
        out = where(masked_rows, zeros, softmax(scores))

    A model that attends over a fixed-size input and passes no attention mask
    can never produce such a row, so the guard never fires. It is not free: the
    compare, the two logical_nots, the reduction, the zeros and the select all
    run over the full attention matrix. Removing it measured 5.96% of GPU time
    on rfdetr_keypoint and 4.35% on rfdetr_nano_segmentation (Mali-G76), with
    bit-identical outputs.

    The override has to travel with the decomposition that *produces* the guard:
    SDPA is still a single node after ``run_decompositions``, so a table keyed on
    ``_safe_softmax`` alone never fires and yields a byte-identical file.

    ONLY valid for a model that cannot produce an all -inf attention row. Do not
    use it for anything that attends over padding.

    :returns: a table to hand to ``ExportedProgram.run_decompositions``.
    """
    import torch
    from torch._decomp import core_aten_decompositions

    core = core_aten_decompositions()
    attention_ops = [
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten._scaled_dot_product_attention_math.default,
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
    ]
    table = {op: core[op] for op in attention_ops if op in core}
    table[torch.ops.aten._safe_softmax.default] = _plain_softmax
    return table
