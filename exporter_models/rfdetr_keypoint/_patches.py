# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Monkey-patches for making RF-DETR compatible with torch.export.

Usage:
    from rfdetr.export._executorch.export_patches import apply_patches
    apply_patches()
    # now import and use rfdetr normally
"""

from typing import Optional

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _precompute_proposals_validity(spatial_shapes_list):
    """Return (all_valid: bool, valid_np: np.ndarray[bool, shape (S,)]).

    Computes output_proposals_valid using concrete Python math so that no comparison ops
    (gt/lt/all/logical_not) are emitted into the exported graph. For RF-DETR with a fixed-size
    input all proposals are always valid, so the masked_fill calls can be removed entirely.
    """
    valid_list = []
    for lvl, (h_, w_) in enumerate(spatial_shapes_list):
        gx = (np.arange(w_, dtype=np.float64) + 0.5) / w_
        gy = (np.arange(h_, dtype=np.float64) + 0.5) / h_
        gy_grid, gx_grid = np.meshgrid(gy, gx, indexing="ij")
        w = h = 0.05 * (2.0**lvl)
        props = np.stack(
            [
                gx_grid,
                gy_grid,
                np.full((h_, w_), w, dtype=np.float64),
                np.full((h_, w_), h, dtype=np.float64),
            ],
            axis=-1,
        )
        valid_list.append(((props > 0.01) & (props < 0.99)).all(axis=-1).reshape(-1))
    valid_np = np.concatenate(valid_list, axis=0)
    return bool(valid_np.all()), valid_np


def _manual_bilinear_grid_sample(input, grid, padding_mode="zeros", align_corners=False):
    """Gather-based bilinear ``grid_sample`` with no ``aten.grid_sampler_2d``.

    Numerically identical to
    ``F.grid_sample(input, grid, mode="bilinear", padding_mode=padding_mode,
    align_corners=align_corners)``, but built only from gather + arithmetic — all
    ops the MLX delegate can lower. This is the MPS path already shipped in
    :func:`rfdetr.utilities.tensors._bilinear_grid_sample`, lifted out of its
    ``device.type == "mps"`` gate so it also runs during CPU-traced export (where
    that helper would otherwise fall through to ``F.grid_sample``).

    ``input`` is ``(N, C, H, W)``; ``grid`` is ``(N, Hg, Wg, 2)`` in ``[-1, 1]``;
    returns ``(N, C, Hg, Wg)``.
    """
    batch_size, channels, height, width = input.shape
    grid_height, grid_width = grid.shape[1], grid.shape[2]

    if align_corners:
        ix = (grid[..., 0] + 1) * (width - 1) / 2
        iy = (grid[..., 1] + 1) * (height - 1) / 2
    else:
        ix = (grid[..., 0] + 1) * width / 2 - 0.5
        iy = (grid[..., 1] + 1) * height / 2 - 0.5

    ix0 = ix.floor().long()
    iy0 = iy.floor().long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # Fractional distance from the top-left corner, cast to the input dtype so
    # float16/bfloat16 inputs don't silently upcast.
    wx1 = (ix - ix0.float()).to(input.dtype).unsqueeze(1)
    wy1 = (iy - iy0.float()).to(input.dtype).unsqueeze(1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    if padding_mode == "border":
        ix0 = ix0.clamp(0, width - 1)
        iy0 = iy0.clamp(0, height - 1)
        ix1 = ix1.clamp(0, width - 1)
        iy1 = iy1.clamp(0, height - 1)
    else:  # zeros: remember which corners are in-bounds before clamping for gather
        in_x0 = (ix0 >= 0) & (ix0 < width)
        in_x1 = (ix1 >= 0) & (ix1 < width)
        in_y0 = (iy0 >= 0) & (iy0 < height)
        in_y1 = (iy1 >= 0) & (iy1 < height)
        ix0 = ix0.clamp(0, width - 1)
        iy0 = iy0.clamp(0, height - 1)
        ix1 = ix1.clamp(0, width - 1)
        iy1 = iy1.clamp(0, height - 1)

    flat = input.flatten(2)  # (N, C, H*W)

    def _gather(iy_, ix_):
        idx = (iy_ * width + ix_).flatten(1).unsqueeze(1).expand(batch_size, channels, -1)
        return flat.gather(2, idx).view(batch_size, channels, grid_height, grid_width)

    v00 = _gather(iy0, ix0)  # top-left
    v10 = _gather(iy0, ix1)  # top-right
    v01 = _gather(iy1, ix0)  # bottom-left
    v11 = _gather(iy1, ix1)  # bottom-right

    if padding_mode == "zeros":
        # Cast masks to the input dtype so the multiply stays float*float (some
        # delegates choke on bool*float type promotion).
        v00 = v00 * (in_x0 & in_y0).to(input.dtype).unsqueeze(1)
        v10 = v10 * (in_x1 & in_y0).to(input.dtype).unsqueeze(1)
        v01 = v01 * (in_x0 & in_y1).to(input.dtype).unsqueeze(1)
        v11 = v11 * (in_x1 & in_y1).to(input.dtype).unsqueeze(1)

    return wx0 * wy0 * v00 + wx1 * wy0 * v10 + wx0 * wy1 * v01 + wx1 * wy1 * v11


def _ms_deform_attn_core_grid_free(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    value_spatial_shapes_hw=None,
):
    """MLX-friendly deformable-attention core: no ``grid_sample``, no ``split``+``getitem``.

    Same math and shapes as the default
    :func:`rfdetr.models.ops.functions.ms_deform_attn_func.ms_deform_attn_core_pytorch`,
    but rebuilt from ops the MLX delegate can lower:

      * per-level value blocks are sliced with ``narrow`` instead of
        ``value.split(...)[level]`` — the MLX ``getitem`` handler cannot index the
        tuple that ``Tensor.split`` returns (``'Slot' object is not subscriptable``), and
      * bilinear sampling uses :func:`_manual_bilinear_grid_sample` instead of
        ``F.grid_sample`` — the MLX delegate has no ``aten.grid_sampler_2d`` handler.

    ``value_spatial_shapes_hw`` must be Python int pairs (as the default core
    requires) so the per-level ``narrow`` lengths stay static under torch.export.
    """
    batch_size, n_heads, head_dim, _ = value.shape
    _, len_query, _, num_levels, num_points, _ = sampling_locations.shape
    shapes = value_spatial_shapes_hw if value_spatial_shapes_hw is not None else value_spatial_shapes

    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    offset = 0
    for level_index, (height, width) in enumerate(shapes):
        # narrow() instead of value.split(...)[level_index] — avoids split's tuple+getitem.
        value_l_ = value.narrow(3, offset, height * width).reshape(batch_size * n_heads, head_dim, height, width)
        offset += height * width
        # batch_size, len_query, n_heads, num_points, 2 -> batch_size*n_heads, len_query, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_index].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = _manual_bilinear_grid_sample(
            value_l_, sampling_grid_l_, padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * n_heads, 1, len_query, num_levels * num_points
    )
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(batch_size, n_heads * head_dim, len_query)
    return output.transpose(1, 2).contiguous()


def _gen_sineembed_for_position_compile_safe(pos_tensor, dim: int = 128):
    """Compile-safe ``gen_sineembed_for_position`` for the MLX delegate.

    Rewrites the ``dim_t`` frequency term to drop two ops the MLX delegate dislikes:

      * ``aten.floor_divide`` (``dim_t // 2``) -> ``(dim_t / 2).floor()``. floor_divide
        forces the whole delegate into the slow interpreter (see
        ``program_is_compile_safe`` in the MLX runtime); ``floor`` is compile-safe.
      * ``aten.pow.Scalar`` (``10000 ** x``) -> ``exp(x * ln(10000))``. pow.Scalar has
        no MLX handler (CPU fallback + delegate split); ``exp``/``mul`` are supported.

    Both stay inside the MLX-compiled graph, and — unlike materializing a
    ``torch.tensor`` constant — this also traces under strict=True (used by the
    quantized path) without a lifted_tensor error. Numerically identical.
    """
    scale = 2 * math.pi
    dim = int(dim)
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = torch.exp((2.0 * (dim_t / 2).floor() / dim) * math.log(10000.0))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


# Populated by ``_shape_as_tensor_traceable`` during the trace: the concrete
# (H, W) of each feature-pyramid level, in level order. ``Transformer.forward``
# builds the ``spatial_shapes`` *tensor* from ``torch._shape_as_tensor(src)`` and
# threads only that tensor (not the Python-int ``spatial_shapes_hw`` list) into
# the keypoint decoder -> MSDeformAttn. Without the static list, the deform-attn
# core iterates the tensor and every ``h * w`` / ``value.split([h*w, ...])``
# emits a ``local_scalar_dense`` (``.item()``) unbacked symint, hundreds of them
# across the decoder layers, which then blow up ExecuTorch's stride comparison
# ("Could not guard on data-dependent expression u… < 1"). We capture the sizes
# here (the one place they exist as concrete ints) and re-inject them below.
_CAPTURED_SPATIAL_SHAPES_HW: list = []


def _shape_as_tensor_traceable(t):
    """Traceable stand-in for ``torch._shape_as_tensor`` (a private ATen op that
    ``torch.export`` refuses to trace -> "data is not allocated yet"). The
    original ``Transformer.forward`` uses it to build ``spatial_shapes`` for
    TensorRT shape-tensor compatibility; for the ExecuTorch/XNNPACK static-shape
    export we bake the concrete sizes as a constant instead. This is the one op
    that blocks the keypoint export while keeping the model's own forward (which
    the whole-forward patches would replace, breaking keypoint code paths).

    ``Transformer.forward`` calls it once per feature-pyramid ``src`` (rank-4
    ``[N, C, H, W]``); we record each level's static ``(H, W)`` so the deform-attn
    patch can consume Python ints instead of tensor scalars (see
    ``_CAPTURED_SPATIAL_SHAPES_HW``)."""
    import torch
    if t.dim() == 4:
        _CAPTURED_SPATIAL_SHAPES_HW.append((int(t.shape[2]), int(t.shape[3])))
    return torch.tensor([int(s) for s in t.shape], dtype=torch.int64, device=t.device)


def apply_patches(
    transformers_import: bool = True,
    proposals: bool = True,
    transformer_forward: bool = False,
    decoder_forward: bool = False,
    decoder_layer_forward: bool = False,
    ms_deform_attn_forward: bool = True,
    ms_deform_attn_core: bool = True,
    ms_deform_attn_grid_free: bool = False,
    pos_embed_compile_safe: bool = False,
    shape_as_tensor_safe: bool = True,
):
    """Apply patches needed for torch.export compatibility.

    Each patch is opt-in via a keyword flag so we can isolate which ones the
    keypoint model needs vs. which conflict with its keypoint-specific control
    flow (e.g. ``cross_attn_srcs`` threading the original
    ``transformer_forward`` patch doesn't know about).

    Defaults are tuned for the keypoint variant: the assert/iteration patches
    that are pure refactors stay on; the patches that replace whole forward
    methods (and thus lose keypoint-specific code paths) stay off.
    """
    if transformers_import:
        _patch_transformers_import()

    if shape_as_tensor_safe:
        import torch
        torch._shape_as_tensor = _shape_as_tensor_traceable

    import rfdetr.models.ops.functions.ms_deform_attn_func as deform_attn_func_mod
    import rfdetr.models.ops.modules.ms_deform_attn as deform_attn_mod
    import rfdetr.models.transformer as transformer_mod

    if proposals:
        transformer_mod.gen_encoder_output_proposals = _gen_encoder_output_proposals
    if transformer_forward:
        transformer_mod.Transformer.forward = _transformer_forward
    if decoder_forward:
        transformer_mod.TransformerDecoder.forward = _transformer_decoder_forward
    if decoder_layer_forward:
        transformer_mod.TransformerDecoderLayer.forward = _decoder_layer_forward
        transformer_mod.TransformerDecoderLayer.forward_post = _decoder_layer_forward_post
    if ms_deform_attn_forward:
        deform_attn_mod.MSDeformAttn.forward = _ms_deform_attn_forward
    if ms_deform_attn_core:
        deform_attn_func_mod.ms_deform_attn_core_pytorch = _ms_deform_attn_core_pytorch
        deform_attn_mod.ms_deform_attn_core_pytorch = _ms_deform_attn_core_pytorch
    if ms_deform_attn_grid_free:
        # MLX path: replace grid_sample + split with gather + narrow so the
        # deformable-attention core lowers fully to the MLX delegate.
        deform_attn_func_mod.ms_deform_attn_core_pytorch = _ms_deform_attn_core_grid_free
        deform_attn_mod.ms_deform_attn_core_pytorch = _ms_deform_attn_core_grid_free
    if pos_embed_compile_safe:
        # MLX path: gen_sineembed_for_position's dim_t is constant; precomputing it
        # drops aten.floor_divide (forces the delegate into the slow interpreter)
        # and aten.pow.Scalar (CPU fallback), letting the delegate run compiled.
        transformer_mod.gen_sineembed_for_position = _gen_sineembed_for_position_compile_safe


def _patch_transformers_import():
    """Fix missing find_pruneable_heads_and_indices in transformers>=5.0."""
    import transformers.pytorch_utils as pu

    if not hasattr(pu, "find_pruneable_heads_and_indices"):

        def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices


def _gen_encoder_output_proposals(
    memory,
    memory_padding_mask,
    spatial_shapes,
    unsigmoid=True,
    spatial_shapes_list=None,
):
    n_, s_, c_ = memory.shape
    proposals = []
    _cur = 0
    _iter = spatial_shapes_list if spatial_shapes_list is not None else spatial_shapes

    if spatial_shapes_list is not None:
        all_valid, valid_np = _precompute_proposals_validity(spatial_shapes_list)
    else:
        all_valid = False
        valid_np = None

    for lvl, (h_, w_) in enumerate(_iter):
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + h_ * w_)].view(n_, h_, w_, 1)
            valid_h = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_w = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            valid_h = torch.full((n_,), h_, dtype=torch.long, device=memory.device)
            valid_w = torch.full((n_,), w_, dtype=torch.long, device=memory.device)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h_, dtype=torch.float32, device=memory.device),
            torch.arange(w_, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).view(n_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(n_, -1, -1, -1) + 0.5) / scale

        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = torch.cat((grid, wh), -1).view(n_, -1, 4)
        proposals.append(proposal)
        _cur += h_ * w_

    output_proposals = torch.cat(proposals, 1)

    if all_valid:
        output_proposals_valid = None
    else:
        if valid_np is not None:
            output_proposals_valid = torch.as_tensor(
                valid_np[:, None], dtype=torch.bool, device=memory.device
            ).unsqueeze(0)
        else:
            output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        if output_proposals_valid is not None:
            output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        if output_proposals_valid is not None:
            output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    if output_proposals_valid is not None:
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


def _transformer_forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None):
    # Fixed-size inference has no padding — mask is always all-False.
    masks = None
    src_flatten = []
    mask_flatten = [] if masks is not None else None
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    valid_ratios = [] if masks is not None else None
    for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
        bs, c, h, w = src.shape
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)

        src = src.flatten(2).transpose(1, 2)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed_flatten.append(pos_embed)
        src_flatten.append(src)
        if masks is not None:
            mask = masks[lvl].flatten(1)
            mask_flatten.append(mask)
    memory = torch.cat(src_flatten, 1)
    if masks is not None:
        mask_flatten = torch.cat(mask_flatten, 1)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

    spatial_shapes_list = list(spatial_shapes)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

    if self.two_stage:
        output_memory, output_proposals = _gen_encoder_output_proposals(
            memory,
            mask_flatten,
            spatial_shapes,
            unsigmoid=not self.bbox_reparam,
            spatial_shapes_list=spatial_shapes_list,
        )
        refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
        group_detr = self.group_detr if self.training else 1
        for g_idx in range(group_detr):
            output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory))

            enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](output_memory_gidx)
            if self.bbox_reparam:
                enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                enc_outputs_coord_cxcy_gidx = (
                    enc_outputs_coord_delta_gidx[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
                )
                enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:]
                enc_outputs_coord_unselected_gidx = torch.concat(
                    [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1
                )
            else:
                enc_outputs_coord_unselected_gidx = self.enc_out_bbox_embed[g_idx](output_memory_gidx) + output_proposals

            topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2])
            topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1]

            refpoint_embed_gidx_undetach = torch.gather(
                enc_outputs_coord_unselected_gidx,
                1,
                topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4),
            )
            refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()

            tgt_undetach_gidx = torch.gather(
                output_memory_gidx,
                1,
                topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model),
            )

            refpoint_embed_ts.append(refpoint_embed_gidx)
            memory_ts.append(tgt_undetach_gidx)
            boxes_ts.append(refpoint_embed_gidx_undetach)
        refpoint_embed_ts = torch.cat(refpoint_embed_ts, dim=1)
        memory_ts = torch.cat(memory_ts, dim=1)
        boxes_ts = torch.cat(boxes_ts, dim=1)

    if self.dec_layers > 0:
        tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)
        refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)
        if self.two_stage:
            ts_len = refpoint_embed_ts.shape[-2]
            refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
            refpoint_embed_subset = refpoint_embed[..., ts_len:, :]

            if self.bbox_reparam:
                refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                refpoint_embed_cxcy = refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
                refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                refpoint_embed_ts_subset = torch.concat([refpoint_embed_cxcy, refpoint_embed_wh], dim=-1)
            else:
                refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts

            refpoint_embed = torch.concat([refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)

        hs, references = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten,
            refpoints_unsigmoid=refpoint_embed,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else valid_ratios,
            spatial_shapes_list=spatial_shapes_list,
        )
    else:
        assert self.two_stage, "if not using decoder, two_stage must be True"
        hs = None
        references = None

    if self.two_stage:
        if self.bbox_reparam:
            return hs, references, memory_ts, boxes_ts
        else:
            return hs, references, memory_ts, boxes_ts.sigmoid()
    return hs, references, None, None


def _transformer_decoder_forward(
    self,
    tgt,
    memory,
    tgt_mask: Optional[Tensor] = None,
    memory_mask: Optional[Tensor] = None,
    tgt_key_padding_mask: Optional[Tensor] = None,
    memory_key_padding_mask: Optional[Tensor] = None,
    pos: Optional[Tensor] = None,
    refpoints_unsigmoid: Optional[Tensor] = None,
    level_start_index: Optional[Tensor] = None,
    spatial_shapes: Optional[Tensor] = None,
    valid_ratios: Optional[Tensor] = None,
    spatial_shapes_list=None,
):
    output = tgt

    intermediate = []
    hs_refpoints_unsigmoid = [refpoints_unsigmoid]

    def get_reference(refpoints):
        obj_center = refpoints[..., :4]
        if self._export:
            from rfdetr.models.transformer import gen_sineembed_for_position

            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model / 2)
            refpoints_input = obj_center[:, :, None]
        else:
            refpoints_input = obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            from rfdetr.models.transformer import gen_sineembed_for_position

            query_sine_embed = gen_sineembed_for_position(refpoints_input[:, :, 0, :], self.d_model / 2)
        query_pos = self.ref_point_head(query_sine_embed)
        return obj_center, refpoints_input, query_pos, query_sine_embed

    if self.lite_refpoint_refine:
        if self.bbox_reparam:
            obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
        else:
            obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

    for layer_id, layer in enumerate(self.layers):
        if not self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

        pos_transformation = 1
        query_pos = query_pos * pos_transformation

        output = layer(
            output,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            query_sine_embed=query_sine_embed,
            is_first=(layer_id == 0),
            reference_points=refpoints_input,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            spatial_shapes_list=spatial_shapes_list,
        )

        if not self.lite_refpoint_refine:
            new_refpoints_delta = self.bbox_embed(output)
            new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
            if layer_id != self.num_layers - 1:
                hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
            refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

        if self.return_intermediate:
            intermediate.append(self.norm(output))

    if self.norm is not None:
        output = self.norm(output)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)

    if self.return_intermediate:
        if self._export:
            hs = intermediate[-1]
            if self.bbox_embed is not None:
                ref = hs_refpoints_unsigmoid[-1]
            else:
                ref = refpoints_unsigmoid
            return hs, ref
        if self.bbox_embed is not None:
            return [torch.stack(intermediate), torch.stack(hs_refpoints_unsigmoid)]
        else:
            return [torch.stack(intermediate), refpoints_unsigmoid.unsqueeze(0)]

    return output.unsqueeze(0), refpoints_unsigmoid.unsqueeze(0)


def _decoder_layer_forward_post(
    self,
    tgt,
    memory,
    tgt_mask=None,
    memory_mask=None,
    tgt_key_padding_mask=None,
    memory_key_padding_mask=None,
    pos=None,
    query_pos=None,
    query_sine_embed=None,
    is_first=False,
    reference_points=None,
    spatial_shapes=None,
    level_start_index=None,
    spatial_shapes_list=None,
):
    bs, num_queries, _ = tgt.shape

    q = k = tgt + query_pos
    v = tgt
    if self.training:
        q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
        k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)
        v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)

    tgt2 = self.self_attn(
        q,
        k,
        v,
        attn_mask=tgt_mask,
        key_padding_mask=tgt_key_padding_mask,
        need_weights=False,
    )[0]

    if self.training:
        tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)

    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)

    tgt2 = self.cross_attn(
        self.with_pos_embed(tgt, query_pos),
        reference_points,
        memory,
        spatial_shapes,
        level_start_index,
        memory_key_padding_mask,
        spatial_shapes_list=spatial_shapes_list,
    )

    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    return tgt


def _decoder_layer_forward(
    self,
    tgt,
    memory,
    tgt_mask=None,
    memory_mask=None,
    tgt_key_padding_mask=None,
    memory_key_padding_mask=None,
    pos=None,
    query_pos=None,
    query_sine_embed=None,
    is_first=False,
    reference_points=None,
    spatial_shapes=None,
    level_start_index=None,
    spatial_shapes_list=None,
):
    return self.forward_post(
        tgt,
        memory,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        pos,
        query_pos,
        query_sine_embed,
        is_first,
        reference_points,
        spatial_shapes,
        level_start_index,
        spatial_shapes_list=spatial_shapes_list,
    )


def _ms_deform_attn_forward(
    self,
    query,
    reference_points,
    input_flatten,
    input_spatial_shapes,
    input_level_start_index,
    input_padding_mask=None,
    input_spatial_shapes_hw=None,
):
    """Rank-5 variant of MSDeformAttn.forward — required for CoreML (rank<=5 cap).

    Collapses ``(n_levels, n_points)`` into a single ``n_levels*n_points`` axis in
    ``sampling_offsets``/``sampling_locations``, paired with a matching rank-5
    ``ms_deform_attn_core_pytorch`` that slices ``[lid*P:(lid+1)*P]`` instead of
    indexing ``[..., lid]``.
    """
    n, len_q, _ = query.shape
    _, len_in, _ = input_flatten.shape
    if not self._export:
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == len_in

    # The keypoint decoder threads only the spatial_shapes *tensor* (not the
    # Python-int list) down here, so fall back to the sizes captured at the
    # torch._shape_as_tensor site. Using ints keeps the per-level narrow/split
    # lengths static and avoids emitting `.item()` unbacked symints that break
    # ExecuTorch lowering. Take the last n_levels in case any stray capture
    # preceded this decoder's feature pyramid.
    if input_spatial_shapes_hw is None and _CAPTURED_SPATIAL_SHAPES_HW:
        input_spatial_shapes_hw = _CAPTURED_SPATIAL_SHAPES_HW[-self.n_levels:]

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))

    sampling_offsets = self.sampling_offsets(query).view(
        n, len_q, self.n_heads, self.n_levels * self.n_points, 2
    )
    attention_weights = self.attention_weights(query).view(
        n, len_q, self.n_heads, self.n_levels * self.n_points
    )

    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        off_norm_exp = (
            offset_normalizer.unsqueeze(1).expand(-1, self.n_points, -1).reshape(self.n_levels * self.n_points, 2)
        )
        ref_pts_exp = (
            reference_points.unsqueeze(3)
            .expand(-1, -1, -1, self.n_points, -1)
            .reshape(n, len_q, self.n_levels * self.n_points, 2)
            .unsqueeze(2)
        )
        sampling_locations = ref_pts_exp + sampling_offsets / off_norm_exp[None, None, None, :, :]
    elif reference_points.shape[-1] == 4:
        ref_center = (
            reference_points[..., :2]
            .unsqueeze(3)
            .expand(-1, -1, -1, self.n_points, -1)
            .reshape(n, len_q, self.n_levels * self.n_points, 2)
            .unsqueeze(2)
        )
        ref_wh = (
            reference_points[..., 2:]
            .unsqueeze(3)
            .expand(-1, -1, -1, self.n_points, -1)
            .reshape(n, len_q, self.n_levels * self.n_points, 2)
            .unsqueeze(2)
        )
        sampling_locations = ref_center + sampling_offsets / self.n_points * ref_wh * 0.5
    else:
        raise ValueError(
            f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
        )

    attention_weights = F.softmax(attention_weights, -1)

    value = value.transpose(1, 2).contiguous().view(n, self.n_heads, self.d_model // self.n_heads, len_in)
    output = _ms_deform_attn_core_pytorch(
        value,
        input_spatial_shapes,
        sampling_locations,
        attention_weights,
        value_spatial_shapes_hw=input_spatial_shapes_hw,
    )
    output = self.output_proj(output)
    return output


def _ms_deform_attn_core_pytorch(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    value_spatial_shapes_hw=None,
):
    """Rank-5 variant matching :func:`_ms_deform_attn_forward`.

    ``sampling_locations`` is shape ``[B, Lq, n_heads, n_levels*n_points, 2]`` here
    (rank 5), so per-level slicing uses ``[lid*P:(lid+1)*P]``.
    """
    b, n_heads, head_dim, _ = value.shape
    _, len_q, _, lp, _ = sampling_locations.shape
    _iter = value_spatial_shapes_hw if value_spatial_shapes_hw is not None else value_spatial_shapes
    levels = len(_iter)
    points = lp // levels

    value_list = value.split([h * w for h, w in _iter], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (h, w) in enumerate(_iter):
        value_l_ = value_list[lid_].view(b * n_heads, head_dim, h, w)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_ * points : (lid_ + 1) * points, :].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(b * n_heads, 1, len_q, levels * points)
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(b, n_heads * head_dim, len_q)
    return output.transpose(1, 2).contiguous()
