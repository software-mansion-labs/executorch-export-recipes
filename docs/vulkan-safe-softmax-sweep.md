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

| model | guard | measured after the fix | shipped |
|---|---|---|---|
| rfdetr_nano_detector | 14 | 534.9 -> 500.9 ms, **6.34%**, 5/5 rounds | yes |
| sdxs_512_dreamshaper | 24 | denoise 0.02% (2/5), encode 1.06% (5/5) | no |

**sdxs was measured and rejected.** The prediction that 24 guards times a
per-diffusion-step UNet would compound was wrong. `denoise` gained 0.02% across
2 of 5 rounds against a 120 ms round-to-round spread, i.e. nothing; `encode`
gained 2.4 ms once per generation. The guard costs in proportion to the
attention matrix, and sdxs attends over a 64x64 latent, so it is negligible
beside the UNet convolutions. RF-DETR attends over 577-2308 tokens, which is why
all three RF-DETR models land at 4-6%. **Guard count is not a proxy for gain;
attention size is.** The export keeps the change so it lands free on the next
re-export, but an 880 MB artifact was not republished for it.

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
