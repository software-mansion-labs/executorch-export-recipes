# Which Vulkan exports carry the dead `_safe_softmax` guard

Swept 2026-09-17 over every published Vulkan `.pte`, one representative per
model, 36 targets. Raw counts in `vulkan-safe-softmax-sweep-2026-09-17.json`,
script in `vulkan_guard_sweep.py`.

## Method

The Vulkan serializer writes each op's name as a literal string into the
delegate blob, so `strings <pte> | grep -c '^aten.where.self$'` counts the ops
inside the delegate without a device, a runner or a re-export. The guard is the
co-occurrence of `eq.Scalar` + `any.dim` + `full_like` + `where.self`; the count
reported is the minimum across the four.

Validated against a known pair before trusting it: the rfdetr_nano_segmentation
build before the fix reads 16 guards / 225 views, after it reads 0 / 193, and
that 32-view delta matches what the on-device probe measured independently.

Always compare the downloaded size against the tree API. Two targets 404'd to
0-byte files and one truncated twice (143.6 and 148.4 MB of 175.8), and all
three "parsed clean" - a stub file has no ops in it either.

## Result: 2 of 36

| model | guard | note |
|---|---|---|
| sdxs_512_dreamshaper | 24 | one per UNet attention block, and the UNet runs once per diffusion step |
| rfdetr_nano_detector | 14 | the third RF-DETR model; keypoint and segmentation are already fixed |

Everything else is clean, for two different reasons:

- **The embedders already avoid SDPA.** MiniLM, mpnet, CLIP text/image,
  distiluse, paraphrase and multi-qa export with `attn_implementation="eager"`,
  which emits plain matmul+softmax. They show `softmax` nodes and no guard.
- **The CNNs have no attention.** style transfer, selfie, lraspp, easyocr,
  paddleocr, efficientnet, blazeface, ssdlite and the yolo26 families.

whisper, supertonic (all four nets) and lfm2.5-vl are also clean.

## The view counts are NOT an action list

`views` is every `aten.view_copy.default`, and only the same-shape ones are
removable - this cannot tell them apart, so it is an upper bound. paddleocr
layout's 500 and CLIP text's 157 look large but say nothing on their own.
Splitting them needs the on-device probe. The redundant-view fix measured 1.75%
on rfdetr segmentation and inside noise on keypoint, so a count alone does not
justify a re-export.
