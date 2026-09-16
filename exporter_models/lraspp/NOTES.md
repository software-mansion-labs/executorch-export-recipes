# lraspp notes

## Vulkan: the index-map variant is dropped (2026-09-16)

`export.py` can emit an argmax index map (`[1,520,520]` int32) instead of the
score map (`[1,21,520,520]` float32), to cut the GPU readback from 22.7 MB to
1.08 MB per inference. RNE cannot load it: the semantic segmenter spec accepts
only the score map, so the variant fails at load with

    Spec doesn't match any of the provided variants:
     - Variant 'batched':   Method 'forward' output #0: DType mismatch.
     - Variant 'unbatched': Method 'forward' input #0: Rank mismatch.

The trick paid off only on the Core ML ANE, and Core ML does not ship an index
variant either. It was published to the lraspp repo on 2026-09-16 and removed
the same day from `main` and `v0.11.0`; `v0.10.0` never carried it. The
`index_map` branch in `export.py` is now unreachable from config.
