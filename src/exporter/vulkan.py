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
