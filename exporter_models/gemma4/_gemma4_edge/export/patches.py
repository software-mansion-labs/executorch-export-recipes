"""Monkey-patches the Gemma4 export pipeline needs, individually gated.

Every patch here is a narrow, Python-only workaround for something upstream
(PyTorch decomps, ExecuTorch edge-lowering passes, or HuggingFace
`modeling_gemma4`) that does not survive `torch.export` under the shapes our
recipes use. None of them changes model numerics — each is either an identity
under the PTE's input contract or a pure export-plumbing fix.

The source recipes ran all six unconditionally at module-import scope, so
merely importing an export script mutated the PyTorch/ExecuTorch/transformers
process state. Here each one is behind a keyword flag on `apply_patches()`
and is idempotent, so importing this module does nothing at all.

P7, the scalar-clamp patch, is separate
---------------------------------------
`apply_scalar_clamp_patch(module)` is a public function rather than a flag on
`apply_patches()`, because it is inherently per-instance: it reads each
module's `input_min`/`input_max` buffers, and the bounds differ per layer.

It applies only on the HuggingFace-vision-tower path, which is `xnnpack-mm`.
The Vulkan multimodal recipe builds `gemma4.model.vision`'s edge tower, whose
`ClippableLinear` already stores its bounds as Python floats via `set_bounds`,
and HF `Gemma4MultimodalEmbedder` is plain `nn.Linear` + `Gemma4RMSNorm`, so
`vulkan-mm` has zero `Gemma4ClippableLinear` instances to patch and skips it.
`xnnpack-mm` does wrap the HF tower and does apply it.

Getting this wrong is quiet, not loud: `Gemma4ClippableLinear` surfaces
`aten.clamp.Tensor`, which the Vulkan op registry does not implement (only
`aten.clamp.default`), so the partitioner drops every clipped linear onto the
host. On XNNPACK the cost is extra host ops rather than a failure. The rewrite
is eager bit-exact (max |diff| = 0), so applying it when unnecessary is
harmless — it just patches nothing and returns 0.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# Names of the patches that have already been applied in this process. Every
# _pN helper checks-and-inserts here, so a second `apply_patches(...)` call is
# a no-op. This matters most for P3, which *wraps* the upstream function:
# applying it twice would nest the wrapper inside itself.
_applied: set[str] = set()

# Saved originals, so the wrapping patches can be reasoned about (and, in
# principle, undone). Populated on first application only.
_orig_dim_order_from_stride: Optional[Callable[..., Any]] = None
_orig_remove_noop_call: Optional[Callable[..., Any]] = None

# Public patch names, in the order `apply_patches` evaluates them.
P1_ATEN_ITEM_DECOMP = "aten_item_decomp"
P2_REMOVE_NOOP_SYMINT = "remove_noop_symint_output"
P3_DIM_ORDER_SYMINT = "dim_order_symint"
P4_AUDIO_BLOCKED_5D_MASK = "audio_blocked_5d_mask"
P5_PATCH_EMBEDDER_WHERE = "patch_embedder_where"
P6_VISION_CUSTOM_SDPA = "vision_custom_sdpa"


# ---------------------------------------------------------------------------
# P1-P3 — unbacked-SymInt survival for `use_dynamic_kv_slice`.
#
# `Gemma4Attention.use_dynamic_kv_slice` calls `.item()` on `input_pos[-1]` so
# the global (full-attention) layers can slice K/V and the static mask down to
# `effective_len = input_pos[-1] + 1` before SDPA, turning an O(max_seq_len)
# scan into O(N+1). That `.item()` introduces an *unbacked* SymInt, and three
# edge-lowering stages were written assuming static shapes and trip guards on
# that symbol. All three patches below are narrow, Python-only workarounds.
#
# They are imported lazily so that importing this module costs nothing and has
# no side effects — `torch._decomp` and `executorch.exir` are both expensive.
# ---------------------------------------------------------------------------


def _p1_remove_aten_item_decomp() -> bool:
    """Drop `aten.item` from `torch._decomp.decomposition_table`.

    So the FakeTensor `op_impl` (which returns a proper unbacked SymInt) takes
    over from the `_refs.item` decomp (which tries to cast FakeTensor -> Python
    int and raises).
    """
    if P1_ATEN_ITEM_DECOMP in _applied:
        return False

    import torch
    from torch._decomp import decomposition_table as _decomp_table

    _decomp_table.pop(torch.ops.aten.item.default, None)
    _applied.add(P1_ATEN_ITEM_DECOMP)
    return True


def _p2_remove_noop_symint_output() -> bool:
    """Replace `RemoveNoopPass.call` so it also checks OUTPUT-shape staticness.

    Upstream compares `orig_tensor.shape == out_val.shape` for `slice_copy`
    after only checking that the *input* shape is static; with an unbacked
    SymInt in the output shape the tuple-equality compare raises. The body
    below is upstream's `call` with the extra `out_val` staticness gate added
    to the `slice_copy.Tensor` branch.

    NOTE: this is a full replacement, not a delegating wrapper — the original
    body is copied. If ExecuTorch's `remove_noop_pass` changes upstream, this
    copy silently goes stale and must be re-synced.
    """
    global _orig_remove_noop_call
    if P2_REMOVE_NOOP_SYMINT in _applied:
        return False

    import torch
    from executorch.exir.pass_base import PassResult as _PassResult
    from executorch.exir.passes import remove_noop_pass as _noop_mod
    from executorch.exir.passes.remove_noop_pass import _DEQUANT_OPS, eliminate_dq_q

    def _patched_noop_call(self, graph_module):
        dequant_nodes = []
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in (
                torch.ops.aten.to.dtype,
                torch.ops.aten.dropout.default,
                torch.ops.aten.slice_copy.Tensor,
            ):
                continue
            orig_tensor = node.args[0].meta["val"]
            if orig_tensor is node.meta["val"]:
                if node.args[0].target in _DEQUANT_OPS:
                    dequant_nodes += [node.args[0]]
                node.replace_all_uses_with(node.args[0])
                continue
            if node.target == torch.ops.aten.slice_copy.Tensor:
                out_val = node.meta["val"]
                # The added gate: upstream only guards on `orig_tensor`, so an
                # unbacked SymInt in the OUTPUT shape reaches the `==` compare.
                if (
                    all(isinstance(d, int) for d in orig_tensor.size())
                    and all(isinstance(d, int) for d in out_val.size())
                ):
                    if orig_tensor.shape == out_val.shape:
                        if node.args[0].target in _DEQUANT_OPS:
                            dequant_nodes += [node.args[0]]
                        node.replace_all_uses_with(node.args[0])
        graph_module.graph.eliminate_dead_code()
        eliminate_dq_q(graph_module, dequant_nodes)
        graph_module.graph.lint()
        graph_module.graph.eliminate_dead_code()
        return _PassResult(graph_module, True)

    _orig_remove_noop_call = _noop_mod.RemoveNoopPass.call
    _noop_mod.RemoveNoopPass.call = _patched_noop_call
    _applied.add(P2_REMOVE_NOOP_SYMINT)
    return True


def _p3_dim_order_symint() -> bool:
    """Short-circuit `dim_order_from_stride` to contiguous order for SymInt
    strides.

    The default implementation sorts strides with `guard_size_oblivious`, which
    raises on unbacked strides. Returning `tuple(range(len(stride)))` is the
    contiguous dim order, which is what every tensor in the dynamic-KV-slice
    path actually has.
    """
    global _orig_dim_order_from_stride
    if P3_DIM_ORDER_SYMINT in _applied:
        return False

    import torch
    from executorch.exir import tensor as _exir_tensor

    _orig = _exir_tensor.dim_order_from_stride
    _orig_dim_order_from_stride = _orig

    def _patched_dim_order(stride):
        if any(isinstance(s, torch.SymInt) for s in stride):
            return tuple(range(len(stride)))
        return _orig(stride)

    _exir_tensor.dim_order_from_stride = _patched_dim_order
    _applied.add(P3_DIM_ORDER_SYMINT)
    return True


# ---------------------------------------------------------------------------
# P4 — dynamic-shape audio blocked mask.
# ---------------------------------------------------------------------------


def _p4_audio_blocked_5d_mask() -> bool:
    """Replace HF `Gemma4AudioModel._convert_4d_mask_to_blocked_5d` with an
    unfold+diagonal formulation.

    The stock HF impl builds the blocked mask with `arange`+`gather` over
    shape-dependent slices, which does not survive `torch.export` when the mel
    time dim is dynamic (`frames_dim = 48 * Dim("audio_blocks", 1..63)`). The
    unfold+diagonal rewrite is shape-polymorphic. Discovered in
    `bench_audio_dynamic_blocked.py` (dynamic-shape enabler #1); eager parity
    vs HF `get_audio_features` was cosine 1.0 with N_dyn == N_ref, validated in
    `diag_audio_dyn_frames.py` on 2026-04-23.

    Only needed while a recipe wraps the HuggingFace `Gemma4AudioModel`, which
    is what both multimodal recipes do. `gemma4.model.audio`'s edge tower
    builds its blocked mask itself and does not go through this method.
    """
    if P4_AUDIO_BLOCKED_5D_MASK in _applied:
        return False

    import torch
    import torch.nn.functional as F
    from transformers.models.gemma4 import modeling_gemma4 as _modeling_gemma4

    def _convert_4d_mask_to_blocked_5d_patched(
        self, mask_4d: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, seq_len, _ = mask_4d.shape
        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right
        context_size = chunk_size + max_past_horizon + max_future_horizon

        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len

        m = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
        m = F.pad(m, (max_past_horizon, max_future_horizon), value=False)
        m = m.reshape(
            batch_size, 1, num_blocks, chunk_size,
            padded_seq_len + max_past_horizon + max_future_horizon,
        )
        m = m.unfold(-1, context_size, chunk_size)
        m = torch.diagonal(m, offset=0, dim1=2, dim2=4)
        m = m.permute(0, 1, 4, 2, 3)
        return m.contiguous()

    _modeling_gemma4.Gemma4AudioModel._convert_4d_mask_to_blocked_5d = (
        _convert_4d_mask_to_blocked_5d_patched
    )
    _applied.add(P4_AUDIO_BLOCKED_5D_MASK)
    return True


# ---------------------------------------------------------------------------
# P5-P6 — HF vision tower graph slimming.
#
# ADAPTATION NOTE: the source recipe applied these per-instance
# (`_apply_patch_embedder_where_patch(module)` / `_apply_vision_custom_sdpa_patch(module)`
# walked `module.modules()` and bound the replacement to each matching
# instance, returning a count). `apply_patches()` takes no module argument, so
# both are applied at CLASS level here. The replacement bodies only touch
# `self`, and the same function was bound to every instance of the class
# anyway, so class-level patching is behaviourally identical for the whole
# process. The only thing lost is the per-module count that used to be printed.
# ---------------------------------------------------------------------------


def _p5_patch_embedder_where() -> bool:
    """Skip `torch.where(padding_positions, 0, pe)` in
    `Gemma4VisionPatchEmbedder._position_embeddings`.

    For all-False padding the `where` is the identity, and all-False padding is
    the only case the PTE contract admits (the vision method is exported at a
    fixed patch grid with a zeroed `padding_positions` buffer). Baking it out
    removes the patch-embed `where` from the graph.
    Eager parity: max |diff| = 0.
    """
    if P5_PATCH_EMBEDDER_WHERE in _applied:
        return False

    import torch.nn.functional as F
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder

    def position_embeddings_nowhere(self, pixel_position_ids, padding_positions):
        clamped_positions = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(
            clamped_positions, num_classes=self.position_embedding_size
        )
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        return position_embeddings

    Gemma4VisionPatchEmbedder._position_embeddings = position_embeddings_nowhere
    _applied.add(P5_PATCH_EMBEDDER_WHERE)
    return True


def _p6_vision_custom_sdpa() -> bool:
    """Replace `Gemma4VisionAttention.forward` with a
    `torch.ops.llama.custom_sdpa` call — the same fused MHA kernel the text
    decoder uses.

    Removes the eager attention block
    (Q.K^T -> scale -> softmax -> .V) from the vision-encoder graph and avoids
    materializing the [B, H, S, S] attention-weight tensor (~400 MB/layer at
    672x960, S=2520, 16 heads).

      - Vision has num_heads == num_kv_heads (no GQA), so no repeat_kv plumbing.
      - Vision uses self.scaling = 1.0 with the scale absorbed into q_norm and
        k_norm weights; we forward self.scaling verbatim.
      - HF's forward keeps q/k/v in BSHD through q_norm + RoPE and transposes
        after; custom_sdpa expects BSHD, so we skip the transpose entirely.
      - Bidirectional: attention_mask = None, is_causal = False, start_pos = 0.
      - Eager parity: cos = 1.0000000000 on a real photo / random / zeros
        (max |diff| ~ 5e-5 softmax noise).

    This targets the *HuggingFace* vision tower. Recipes built on
    `gemma4.model.vision`'s edge tower use `cfg.use_fused_sdpa` /
    `cfg.use_vulkan_sdpa` instead and must NOT enable this flag.
    """
    if P6_VISION_CUSTOM_SDPA in _applied:
        return False

    import torch
    # Side-effect import: registers `torch.ops.llama.custom_sdpa`. The source
    # recipe did this at module top; here it is deferred so importing this
    # module stays free. Must happen before the replacement forward is ever
    # *called*.
    import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4VisionAttention,
        apply_multidimensional_rope,
    )

    def forward_custom_sdpa(self, hidden_states, position_embeddings=None,
                            attention_mask=None, position_ids=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        q = self.q_proj(hidden_states).view(hidden_shape)
        q = self.q_norm(q)
        q = apply_multidimensional_rope(q, cos, sin, position_ids)

        k = self.k_proj(hidden_states).view(hidden_shape)
        k = self.k_norm(k)
        k = apply_multidimensional_rope(k, cos, sin, position_ids)

        v = self.v_proj(hidden_states).view(hidden_shape)
        v = self.v_norm(v)

        out_bshd = torch.ops.llama.custom_sdpa(
            q.contiguous(), k.contiguous(), v.contiguous(),
            0,          # start_pos
            None,       # attention mask (bidirectional, no mask)
            0.0,        # dropout
            False,      # is_causal
            float(self.scaling),
        )

        attn_output = out_bshd.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    Gemma4VisionAttention.forward = forward_custom_sdpa
    _applied.add(P6_VISION_CUSTOM_SDPA)
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_patches(
    *,
    dynamic_kv_slice: bool = False,
    dynamic_audio: bool = False,
    vision_custom_sdpa: bool = False,
    patch_embedder_where: bool = False,
) -> list[str]:
    """Apply the requested patches. Returns the names of those applied.

    All patches are process-global and idempotent: a name already applied in
    this process is skipped and omitted from the returned list, so the return
    value is "what changed on this call", not "what is active".

    Args:
        dynamic_kv_slice: Enables P1+P2+P3, the unbacked-SymInt trio. Required
            whenever `Gemma4Attention.use_dynamic_kv_slice = True`; harmless
            otherwise, but leave it off so a static-shape recipe fails loudly
            instead of silently relying on patched lowering.
        dynamic_audio: Enables P4. Required when exporting the HuggingFace
            audio tower with a dynamic mel time dim.
        vision_custom_sdpa: Enables P6. HuggingFace vision tower only.
        patch_embedder_where: Enables P5. HuggingFace vision tower only.

    Ordering constraint: P5/P6 patch classes, not instances, so they may be
    applied at any point before the vision module is traced. P1-P3 mutate the
    decomposition table and edge-lowering passes, so they must be applied
    before `to_edge_transform_and_lower` — applying them at the top of an
    export script is the safe habit.
    """
    applied: list[str] = []

    if dynamic_kv_slice:
        if _p1_remove_aten_item_decomp():
            applied.append(P1_ATEN_ITEM_DECOMP)
        if _p2_remove_noop_symint_output():
            applied.append(P2_REMOVE_NOOP_SYMINT)
        if _p3_dim_order_symint():
            applied.append(P3_DIM_ORDER_SYMINT)

    if dynamic_audio:
        if _p4_audio_blocked_5d_mask():
            applied.append(P4_AUDIO_BLOCKED_5D_MASK)

    if patch_embedder_where:
        if _p5_patch_embedder_where():
            applied.append(P5_PATCH_EMBEDDER_WHERE)

    if vision_custom_sdpa:
        if _p6_vision_custom_sdpa():
            applied.append(P6_VISION_CUSTOM_SDPA)

    return applied


def apply_scalar_clamp_patch(module: "nn.Module") -> int:
    """P7 — rebind `Gemma4ClippableLinear.forward` to use Python-float bounds.

    Returns the number of modules patched.

    This one is NOT part of `apply_patches()` because it is inherently
    per-instance: it reads `m.input_min.item()` off each module, and the bounds
    differ per layer. There is nothing to hoist to the class.

    Why it exists: `Gemma4ClippableLinear` stores its clamp bounds as 0-d Tensor
    buffers, so `torch.clamp(x, self.input_min, self.input_max)` traces to
    `aten.clamp.Tensor`. The Vulkan op registry implements only
    `aten.clamp.default`, and the mismatch surfaces as a KeyError inside
    `insert_prepack_nodes` rather than as a partitioning refusal. Reading the
    bounds out as Python floats at patch time produces `aten.clamp.default`.
    Eager-bit-exact: the values are identical, only their graph representation
    changes.

    When you need it: any recipe that traces the HuggingFace vision or audio
    tower. That is `xnnpack-mm`, which wraps the HF vision tower and applies
    this to both towers for exactly this reason.

    When you do not: the Vulkan multimodal recipe builds the *edge* vision
    tower, whose `ClippableLinear` already stores Python floats via
    `set_bounds`, and HF `Gemma4MultimodalEmbedder` contains no
    `Gemma4ClippableLinear` at all. `vulkan-mm` therefore skips this patch.
    """
    import torch
    import torch.nn as nn  # noqa: F401  (annotation only)

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        # No HF gemma4 in this environment, or the class was renamed upstream.
        # Nothing to patch; report zero rather than failing an export that may
        # not touch the HF towers at all.
        return 0

    def _make_forward(imin: float, imax: float, omin: float, omax: float, clipped: bool):
        if not clipped:
            def forward_no_clip(self, hidden_states):
                return self.linear(hidden_states)
            return forward_no_clip

        def forward_scalar(self, hidden_states):
            hidden_states = torch.clamp(hidden_states, imin, imax)
            hidden_states = self.linear(hidden_states)
            hidden_states = torch.clamp(hidden_states, omin, omax)
            return hidden_states
        return forward_scalar

    n = 0
    for m in module.modules():
        if not isinstance(m, Gemma4ClippableLinear):
            continue
        if not m.use_clipped_linears:
            fwd = _make_forward(0.0, 0.0, 0.0, 0.0, False)
        else:
            fwd = _make_forward(
                float(m.input_min.item()),
                float(m.input_max.item()),
                float(m.output_min.item()),
                float(m.output_max.item()),
                True,
            )
        m.forward = fwd.__get__(m, Gemma4ClippableLinear)
        n += 1
    return n
