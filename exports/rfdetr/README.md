
# RF-DETR ExecuTorch Export

Exports RF-DETR variants to `.pte` (ExecuTorch) models delegated to XNNPACK.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
cd exports/rfdetr
uv sync
```

## Usage

```bash
# Export all variants at their default resolution (560)
uv run rfdetr_export.py

# Export a specific variant
uv run rfdetr_export.py --variant nano
uv run rfdetr_export.py --variant nano small

# Export at specific resolution(s)
uv run rfdetr_export.py --variant nano --resolution 560
uv run rfdetr_export.py --variant nano --resolution 560 640

# Export all variants at multiple resolutions
uv run rfdetr_export.py --resolution 560 640
```

## Variants

| Flag | Class | Default Resolution |
|---|---|---|
| `nano` | `RFDETRNano` | 560 |
| `small` | `RFDETRSmall` | 560 |
| `base` | `RFDETRBase` | 560 |
| `large` | `RFDETRLarge` | 560 |

## Outputs

All `.pte` files are written to `OUTPUT_DIR/xnnpack/`:

```
OUTPUT_DIR/xnnpack/rfdetr_<variant>_det_<resolution>_xnnpack_fp32.pte
```

Example:
```
OUTPUT_DIR/xnnpack/rfdetr_nano_det_560_xnnpack_fp32.pte
OUTPUT_DIR/xnnpack/rfdetr_small_det_560_xnnpack_fp32.pte
```

### Output tensors

| Name | Shape | Description |
|---|---|---|
| `boxes_xyxy` | `[Q, 4]` | Bounding boxes in xyxy pixel coords (0 to resolution) |
| `scores` | `[Q]` | Per-detection confidence in [0, 1] |
| `labels` | `[Q]` | COCO class IDs as float32 |

Scale boxes by `orig_w / resolution` and `orig_h / resolution` to map back to the original image.

## Dependencies

| Dependency | Notes |
|---|---|
| `executorch>=1.1.0` | Built with `XNNPACK=ON` |
| `rfdetr>=1.1.0` | RF-DETR model package |
| `torch>=2.3.0` | PyTorch |
| `numpy`, `tabulate` | Utilities |

Always prefix commands with `PATH=".venv/bin:$PATH"` if you need `flatc` on PATH (required for XNNPACK serialization).

## How it works

1. **Load model** — instantiate `RFDETRNano` / `RFDETRSmall` / etc., move to CPU, call `.export()`.
2. **Apply patches** (`export_patches.py`) — monkey-patches deformable attention and transformer for `torch.export` compatibility.
3. **Wrap** — `RFDETRDetectionWrapper` converts cxcywh → xyxy and runs sigmoid+argmax for scores/labels.
4. **Export** — `torch.export.export` with a random fp32 example input.
5. **Lower to XNNPACK** — `to_edge_transform_and_lower` with:
   - `AddmmToLinearTransform` converts `aten.addmm` → `aten.linear` so XNNPACK can fuse HuggingFace-style encoder layers.
   - `XnnpackPartitioner` delegates compute-heavy ops (GEMM, conv, elementwise) to XNNPACK.
   - `_core_aten_ops_exception_list=[aten.linear]` prevents the edge compiler from re-decomposing linears before partitioning.
6. **Serialize** — writes the `.pte` flatbuffer to `OUTPUT_DIR/xnnpack/`.

## Patches (`export_patches.py`)

RF-DETR uses deformable attention with dynamic spatial shapes, which breaks `torch.export`. The patches fix this without touching model weights or changing numerical outputs.

| Patch | Problem solved |
|---|---|
| `_patch_transformers_import` | `transformers>=5.0` removed `find_pruneable_heads_and_indices`; rfdetr imports it |
| `_transformer_forward` (`masks=None`) | Padding masks are all-False for fixed-size input — forcing `None` removes ~10 ms of portable ops |
| `spatial_shapes_list` threading | `torch.export` traces with `FakeTensor`; iterating over or calling `.item()` on a `FakeTensor` fails. A Python list of concrete `(H, W)` tuples is threaded alongside the tensor |
| `_ms_deform_attn_forward` assert removed | Shape assertion is data-dependent and fails during tracing |
| `_gen_encoder_output_proposals` | Replaced dynamic iteration with Python-side precomputation — all proposals are valid for any reasonable fixed input size |
