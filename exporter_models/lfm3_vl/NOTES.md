# lfm3_vl

LiquidAI/LFM3-VL-2B, pinned at revision
`f83d2b0a2296aaa277f394990a9b8b949ddf9691`. Text side: the LFM3-1.7B
architecture (see `lfm3_text/NOTES.md`; the decoder code is a copy) with its
own weights under `model.language_model.`. Vision side: a 14-layer hybrid
tower (gated depthwise 3x3 convs + bidirectional GQA attention with axial
RoPE), patch 16, 2x2 merge, then a LayerNorm/GELU projector.

## Contract (RNE multimodal runner)

```
vision_encoder  f32[1, 3, 512, 512] raw RGB 0..255 -> f32[1, 256, 2048]
token_embedding i64[1, 1..256]                     -> f32[1, S, 2048]
text_decoder    f32[1, 1..256, 2048], i64[S]       -> f32[1, 128000]
```

Context 8192 (LFM2.5-VL ships 2048), prefill chunk 256 so one image prefills
in one call. In the prompt, an image is `<|vision_start|>`, 256 image
embeddings, `<|vision_end|>` (the upstream processor expands `<|image|>` to
256 placeholder tokens; RNE splices the encoder output at that point).

## Vision tower (`_lfm3/vision.py`)

Fixed 512x512, so the patch grid is 32x32 and the output 256 tokens.
Exact against upstream `get_image_features` in fp32 (`vision_parity.py`:
per-token cosine 1.000000, max abs error 5.5e-6). Three rewrites, each exact
up to rounding:

- rescale + normalize (x/255, then (x - 0.5)/0.5) is in the graph;
- an image is one frame that upstream duplicates along the temporal patch
  axis; the two temporal slices of the patchifier weight are summed instead
  and the patchifier is a stride-16 conv;
- tokens stay in raster order through the tower and are reordered into the
  upstream 2x2 merge-block order only before the projector (attention is
  bidirectional and RoPE uses each token's own (h, w)).

The tower stays unquantized, as in LFM2.5-VL.

## Results

`validate.py` drives the methods the way RNE does, against `reference.py`
(fp32 HF, the docs "SALE NOW ON" banner photo at 512x512, "Describe this
image in one sentence.", thinking off). Eager vendored pipeline: cosine
0.99999, 24/24.

| Variant | Size | Delegates / host ops | cos @256 / @276 | teacher-forced |
| --- | ---: | --- | --- | ---: |
| XNNPACK 8da4w | 2.01 GB | vision 56, decoder 350 / RMSNorm, RoPE | 0.975 / 0.994 | 18/24 |
| MLX int4 | 1.31 GB | 1 per method / none | 0.972 / 0.993 | 17/24 |
| MLX int8 | 2.17 GB | 1 per method / none | 0.9996 / 0.9999 | 24/24 |

Free-running greedy output is correct for every variant; the teacher-forced
misses are wording (e.g. XNNPACK: "Three white banners with red lettering
that says "SALE NOW ON" are displayed on poles in front of a building.").
The XNNPACK file is large because the vision tower is fp32 (~0.8 GB).

Vulkan: same blocker as `lfm3_text` (pytorch/executorch#23539).
