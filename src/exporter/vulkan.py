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

    Handles the BERT and MPNet attention modules. Returns the number of modules
    patched, and raises if that is zero: a transformers change that renames
    these classes would otherwise silently cost the speedup.
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

    patched = 0
    for module in model.modules():
        name = type(module).__name__
        if name == "BertSelfAttention":
            module.forward = types.MethodType(bert_forward, module)
            patched += 1
        elif name == "MPNetSelfAttention":
            module.forward = types.MethodType(mpnet_forward, module)
            patched += 1
    if patched == 0:
        raise RuntimeError(
            "use_rank3_attention patched nothing; the attention module names "
            "changed and the rewrite would silently be a no-op"
        )
    return patched
