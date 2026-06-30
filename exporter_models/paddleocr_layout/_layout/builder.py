"""PP-DocLayoutV3 (RT-DETR document layout detector) — harness-facing exporter.

Builds ONE fused .pte with a single static `forward` method for a given backend.
Ported from the standalone `export_pp_layout.py`; the argparse / labels-emission
shell is gone — the harness authors config.json + labels.json and calls
`build_and_lower(backend, out_path)`.

  in  [1,3,800,800] f32 [0,1] RGB (no mean/std; /255 done by the runtime)
  out (objectDetection contract — RT-DETR decode baked in-graph):
        boxes  [300,4] f32  xyxy in model-input(800) pixels
        scores [300]   f32  = max_c sigmoid(logits)
        classes[300]   f32  = argmax_c logits (float)
      all 300 queries kept (NMS-free; client thresholds). Client does NOT decode.

Per-backend precision (smallest build that keeps ALL boxes — gate = recall@IoU0.3):
  xnnpack : fp32  (int8/int4 lose whole boxes — keep fp32, CPU fallback)
  coreml  : fp16  (ANE; ~41/42 boxes @IoU0.3, ~8px shift)
  vulkan  : fp16  (Android GPU; lossless 42/42; box heads kept on XNNPACK CPU)

The critical tricks (do NOT drop any — see header comments):
  1. force_conv_contiguous — EAGER forward_pre_hook .contiguous() (deformable-attn
     feeds non-contiguous views into convs -> XNNPACK silently garbages on REAL
     images). XNNPACK needs Conv2d; CoreML needs Linear+LayerNorm too.
  2. Trace + verify on a REAL image (layout_verify.png). Noise elides the copies.
  3. _clone_shared_weights + _patch_shared_weight_deletion — RT-DETR ties
     class_embed/bbox_embed across decoder layers -> ET's repeated `del` KeyErrors.
  4. CoreML: patch_sincos_outer_for_coreml + patch_deformable_attention_rank5.
  5. Vulkan: mark_bbox_heads_for_cpu (the 256->4 box heads mis-compute on Xclipse
     -> keep just those on CPU via the _BBoxHeadCPU marker; mixed-delegate to
     XNNPACK so they run the fast CPU kernel, not the slow portable one).
"""
import copy
import inspect
import os
import textwrap

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lowering import lower_save as _lower_save
from .backends import xnnpack, coreml, vulkan

HERE = os.path.dirname(os.path.abspath(__file__))
RID = "PaddlePaddle/PP-DocLayoutV3_safetensors"
HW = 800
VERIFY_IMG = os.path.join(HERE, "layout_verify.png")


# ── ExecuTorch shared-weight `del` bug (belt: clone; braces: tolerant del) ──
_patched = False


def _patch_shared_weight_deletion():
    """Make _unsafe_adjust_original_program tolerate the repeated del of a weight
    shared across delegate partitions (class_embed/bbox_embed reused each decoder
    layer). Source-patch `del state_dict[k]` -> `state_dict.pop(k, None)`, then
    rebind the name in backend_api (it imported it by value)."""
    global _patched
    if _patched:
        return
    import executorch.exir.lowered_backend_module as _lbm
    src = textwrap.dedent(inspect.getsource(_lbm._unsafe_adjust_original_program))
    src = src.replace("del original_program._state_dict[input_target]",
                      "original_program._state_dict.pop(input_target, None)")
    exec(compile(src, _lbm.__file__, "exec"), _lbm.__dict__)
    import executorch.exir.backend.backend_api as _bapi
    _bapi._unsafe_adjust_original_program = _lbm._unsafe_adjust_original_program
    _patched = True


def force_conv_contiguous(model, module_types=None):
    """EAGER forward_pre_hook .contiguous() before selected modules (trick #1).
    XNNPACK only needs Conv2d; CoreML needs Linear+LayerNorm too."""
    if module_types is None:
        module_types = (nn.Conv2d,)

    def pre_hook(mod, args):
        if args and isinstance(args[0], torch.Tensor):
            return (args[0].contiguous(),) + tuple(args[1:])
        return None
    n = 0
    for mod in model.modules():
        if isinstance(mod, module_types):
            mod.register_forward_pre_hook(pre_hook)
            n += 1
    return n


def patch_sincos_outer_for_coreml():
    """Replace `.outer()` in the 2D sincos pos-embed with fp32 broadcast mul —
    coremltools lowers aten.outer to matmul and the int32 arange vs fp32 omega
    operands trip an int32xfp32 dtype-mismatch crash."""
    import transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 as M

    def _build(height, width, embed_dim=256, temperature=10000.0,
               cls_token=False, device=None, dtype=torch.float32):
        if embed_dim % 4 != 0:
            raise ValueError(f"`embed_dim` must be divisible by 4, got {embed_dim}")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / temperature ** omega
        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        emb_h = grid_h.flatten().unsqueeze(1) * omega.unsqueeze(0)
        emb_w = grid_w.flatten().unsqueeze(1) * omega.unsqueeze(0)
        pos_embed = torch.cat([emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1)
        if cls_token:
            pos_embed = torch.cat(
                [torch.zeros(1, embed_dim, dtype=torch.float32, device=device), pos_embed], dim=0)
        return pos_embed.to(dtype)
    M.build_2d_sinusoidal_position_embedding = _build


def patch_deformable_attention_rank5():
    """Rewrite MS-deformable-attention so no rank-6 tensor is built (CoreML max
    rank 5). Computes sampling locations + grid_sample PER LEVEL; numerically
    identical to the stock formula, reordered."""
    import transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 as M

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, position_embeddings=None,
                reference_points=None, spatial_shapes=None, spatial_shapes_list=None,
                level_start_index=None, **kwargs):
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings
        bs, num_queries, _ = hidden_states.shape
        _, seq_len, _ = encoder_hidden_states.shape
        head_dim = self.d_model // self.n_heads
        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(bs, seq_len, self.n_heads, head_dim)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            bs, num_queries, self.n_heads, self.n_levels, self.n_points * 2)
        attention_weights = self.attention_weights(hidden_states).view(
            bs, num_queries, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            bs, num_queries, self.n_heads, self.n_levels, self.n_points)
        num_coordinates = reference_points.shape[-1]
        value_list = value.split([h * w for h, w in spatial_shapes_list], dim=1)
        ref_shared = reference_points.shape[2] == 1
        sampling_value_list = []
        for lvl, (h, w) in enumerate(spatial_shapes_list):
            off_l = sampling_offsets[:, :, :, lvl, :].view(
                bs, num_queries, self.n_heads, self.n_points, 2)
            ref_l = reference_points[:, :, 0 if ref_shared else lvl, :]
            if num_coordinates == 2:
                norm_l = off_l.new_tensor([w, h])
                samp_l = ref_l[:, :, None, None, :] + off_l / norm_l
            else:
                samp_l = (ref_l[:, :, None, None, :2]
                          + off_l / self.n_points * ref_l[:, :, None, None, 2:] * 0.5)
            grid_l = (2 * samp_l - 1).permute(0, 2, 1, 3, 4).reshape(
                bs * self.n_heads, num_queries, self.n_points, 2)
            value_l = (value_list[lvl].flatten(2).transpose(1, 2)
                       .reshape(bs * self.n_heads, head_dim, h, w))
            samp_v = F.grid_sample(value_l, grid_l, mode="bilinear",
                                   padding_mode="zeros", align_corners=False)
            sampling_value_list.append(samp_v)
        attn_w = attention_weights.permute(0, 2, 1, 3, 4).reshape(
            bs * self.n_heads, 1, num_queries, self.n_levels * self.n_points)
        output = ((torch.stack(sampling_value_list, dim=-2).flatten(-2) * attn_w)
                  .sum(-1).view(bs, self.n_heads * head_dim, num_queries)
                  .transpose(1, 2).contiguous())
        return self.output_proj(output), attention_weights
    M.PPDocLayoutV3MultiscaleDeformableAttention.forward = forward


def mark_bbox_heads_for_cpu(model):
    """Vulkan/Xclipse-940 mis-computes the box head (MLP ending Linear 256->4)
    on-device — it returns the 256-wide input, not 4 box coords. Re-tag only the
    two 256->4 box heads with a marker subclass so the nn_module_blocklist keeps
    just those on CPU. Returns the marker typename to blocklist."""
    import transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 as M

    class _BBoxHeadCPU(M.PPDocLayoutV3MLPPredictionHead):
        pass
    n = 0
    for mod in model.modules():
        if (isinstance(mod, M.PPDocLayoutV3MLPPredictionHead)
                and isinstance(mod.layers[-1], nn.Linear)
                and mod.layers[-1].out_features == 4):
            mod.__class__ = _BBoxHeadCPU       # nn_module_stack now records the marker
            n += 1
    print(f"  bbox heads tagged for CPU: {n} (others stay on GPU)")
    return "_BBoxHeadCPU"


def _clone_shared_weights(model):
    """Deep-copy modules whose Parameters share identity (RT-DETR ties
    class_embed/bbox_embed) so each delegate partition owns its own state_dict
    entry — the root fix for the shared-weight del bug."""
    seen, shared = {}, []
    for mod_name, mod in model.named_modules():
        for pname, param in mod._parameters.items():
            if param is None:
                continue
            full = f"{mod_name}.{pname}" if mod_name else pname
            if id(param) in seen:
                shared.append((seen[id(param)], full))
            else:
                seen[id(param)] = full
    if not shared:
        return model
    print(f"  shared param pairs: {len(shared)}")
    to_replace = {name.rsplit('.', 1)[0] if '.' in name else '' for _, name in shared}
    for path in to_replace:
        parts = path.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], copy.deepcopy(getattr(parent, parts[-1])))
    return model


class LayoutDetector(nn.Module):
    """Bakes the RT-DETR decode in-graph so layout speaks the SHARED
    objectDetection contract -> (boxes[Q,4] xyxy in model-input pixels, scores[Q],
    classes[Q] float). All Q=300 queries kept (NMS-free; client thresholds). This
    moves the per-query sigmoid/argmax/cxcywh->xyxy off the client."""

    def __init__(self, model, size=HW):
        super().__init__()
        self.model = model
        self.size = float(size)

    def forward(self, pixel_values):
        out = self.model(pixel_values)
        logits, boxes = out.logits[0], out.pred_boxes[0]       # [Q,C], [Q,4] cxcywh-norm
        scores = torch.sigmoid(logits).amax(-1)                # [Q] = max_c sigmoid
        classes = logits.argmax(-1).to(torch.float32)          # [Q] = argmax_c (float)
        cx, cy, w, h = boxes.unbind(-1)
        xyxy = torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                            cx + 0.5 * w, cy + 0.5 * h], -1) * self.size   # [Q,4] px
        return xyxy, scores, classes


def _real_image_tensor(img_path, size):
    """Preprocess a real image as inference does (resize/255, contiguous NCHW)."""
    from PIL import Image
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def build_and_lower(backend, out_path):
    """Build the PP-DocLayoutV3 PTE (single static `forward`) for `backend` and
    write it to absolute `out_path`. xnnpack fp32 / coreml fp16 / vulkan fp16."""
    assert os.path.isabs(out_path), "out_path must be absolute"
    assert backend in ("xnnpack", "coreml", "vulkan"), f"unknown backend '{backend}'"
    hw = HW

    from transformers import AutoModelForObjectDetection
    model = AutoModelForObjectDetection.from_pretrained(RID).eval()
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    if backend == "coreml":
        patch_sincos_outer_for_coreml()   # all-fp32 sincos (avoids float64 -> half mismatch)
        patch_deformable_attention_rank5()

    model = _clone_shared_weights(model)

    mod_types = ((nn.Conv2d, nn.Linear, nn.LayerNorm) if backend == "coreml"
                 else (nn.Conv2d,))
    n = force_conv_contiguous(model, mod_types)
    print(f"  force_conv_contiguous: {n} {'/'.join(t.__name__ for t in mod_types)}")

    vk_blocklist = None
    if backend == "vulkan":
        vk_blocklist = [mark_bbox_heads_for_cpu(model)]

    wrapper = LayoutDetector(model, size=hw).eval()

    # trace on a REAL image — noise makes the contiguity fix elide (trick #2).
    if os.path.exists(VERIFY_IMG):
        x = _real_image_tensor(VERIFY_IMG, hw)
        print(f"  trace input: real image {os.path.basename(VERIFY_IMG)}")
    else:
        x = torch.randn(1, 3, hw, hw)
        print("  ⚠ trace input: SYNTHETIC noise (no image) — fix may not land")

    print("[1/3] torch.export...")
    with torch.no_grad():
        ep = torch.export.export(wrapper, (x,), strict=False)

    print(f"[2/3] lower ({backend})...")
    _patch_shared_weight_deletion()
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.passes import MemoryPlanningPass

    if backend == "coreml":
        partitioner = coreml.layout_partitioner("float16", "none")
        passes = []
        cc = EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)
    elif backend == "vulkan":
        # MIXED-DELEGATE: Vulkan fp16 for the net; the two 256->4 box heads stay on
        # CPU (Xclipse mis-computes them) and run the FAST XNNPACK kernel via the
        # appended partitioner (addmm->linear so XNNPACK claims those matmuls).
        from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
        partitioner = [vulkan.partitioner(force_fp16=True, nn_module_blocklist=vk_blocklist),
                       xnnpack.partitioner()]
        passes = [AddmmToLinearTransform()]
        cc = EdgeCompileConfig(_check_ir_validity=False,
                               _core_aten_ops_exception_list=[torch.ops.aten.linear.default])
    else:  # xnnpack fp32 (bit-exact; int8/int4 lose whole boxes)
        from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
        partitioner = xnnpack.layout_partitioner("none")
        passes = [AddmmToLinearTransform()]
        cc = EdgeCompileConfig(_check_ir_validity=False,
                               _core_aten_ops_exception_list=[torch.ops.aten.linear.default])

    parts = partitioner if isinstance(partitioner, list) else [partitioner]
    print("[3/3] lower + serialize...")
    out = _lower_save({"forward": ep}, parts, out_path, transform_passes=passes,
                      compile_config=cc,
                      memory_planning=MemoryPlanningPass(alloc_graph_input=False), width=10)
    return out
