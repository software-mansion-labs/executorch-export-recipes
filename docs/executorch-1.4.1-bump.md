# ExecuTorch 1.4.1 bump

Record of moving every per-model export environment from the previous
1.3.1 / 1.2.0 / nightly mix onto ExecuTorch 1.4.1.

## What the runtime side pins

| Piece | Where |
|---|---|
| ExecuTorch fork | `software-mansion-labs/executorch` @ `@ms/separate-backends-1.4.1` (v1.4.1 + our fixes) |
| Tokenizers | `software-mansion-labs/pytorch-tokenizers` @ `@ms/build-1.4.1` |
| Prebuilt native libs | RNE GitHub pre-release `v0.10.0-libs-1.4.1` (runtime only: Android `.so`, iOS xcframeworks, headers) |

Only 3 of the 19 fork commits over `v1.4.1` touch the ahead-of-time export path,
and all three are Vulkan or MLX (`backends/vulkan/serialization/vulkan_graph_builder.py`,
`backends/vulkan/op_registry.py`, `backends/mlx/ops.py`). **Nothing in the fork changes
XNNPACK or CoreML lowering.** That is why XNNPACK/CoreML models pin the PyPI wheel and
only `gemma4`, `gemma4_omni` and `lfm2_5_vl` need the fork source build.

## Re-exporting is a no-op for the XNNPACK path

Controlled test — same model source, same torch, *only* the `executorch` pin changed —
produces **byte-identical** `.pte` on 1.3.1 and 1.4.1:

| Model | Artifact | SHA-256 (1.3.1 == 1.4.1) |
|---|---|---|
| `all_minilm_l6_v2` | `all_minilm_l6_v2_xnnpack_fp32.pte` | `3e59e86a…` |
| `style_transfer_candy` | `style_transfer_candy_xnnpack_{fp32,int8}.pte` | `8320ded6…` |
| `efficientnet_v2_s` | `efficientnet_v2_s_xnnpack_{fp32,int8}.pte` | `eb0cdbe2…` |
| `fsmn_vad` | `fsmn_vad_xnnpack_fp32.pte` | `9158ab9c…` |
| `selfie_segmentation` | `selfie_segmentation_xnnpack_fp32.pte` | `ea5cd127…` |
| `selfie_segmentation` | `selfie_segmentation_landscape_xnnpack_fp32.pte` | `ea70b56f…` |

**So do not re-export and re-publish plain XNNPACK models just because the ET pin moved.**
The value of this bump is keeping the export env aligned with the 1.4.1 runtime, not new
artifacts. Before re-exporting anything, hash the old and new `.pte` and skip the ones
that match. Re-export is expected to matter only where MLX or Vulkan is involved.

To reproduce the comparison for a model: export it once on the current pins, once with
the pins from the commit before the bump, and `shasum -a 256` both.

## Version-specific traps hit during the bump

- **`torchao` `.dev` sorts below the release.** 1.4.1 requires `torchao>=0.18.0`, and
  `0.18.0.dev20260702` is *less than* `0.18.0`, so 10 model locks became unsatisfiable at
  once with an identical error. Fix: pin `torchao==0.18.0` and delete the
  `torchao = { index = "pytorch-nightly" }` source line.
- **`whisper` was pinned to 1.2.0 because its CoreML export failed on 1.3.1.** It exports
  cleanly on 1.4.1, so the hold is lifted.
- **`lfm2_5_text` was pinned to an ExecuTorch nightly** because the 1.3.1 wheel's
  `_portable_lib` was ABI-incompatible with the torch 2.14 nightly (missing
  `c10::impl::cow::materialize_cow_storage`). The 1.4.1 release wheel loads fine against
  `torch==2.14.0.dev20260702`, so it now uses the stable wheel and the standard torch pin.

## CoreML `.pte` cannot be verified in-process

Loading a CoreML `.pte` through the pip wheel's pybindings fails with:

```
[ETCoreMLModelManager.mm] Multifunction CoreML models require iOS 18.0+ / macOS 15.0+ SDK to build.
[method.cpp:132] Init failed for backend CoreMLBackend: 0x23
```

This is the wheel's prebuilt delegate, **not** a 1.4.1 export regression: a
1.2.0-produced artifact fails identically under the 1.4.1 runtime. CoreML verification
needs a physical device.

## Pre-existing issues found while verifying (NOT caused by this bump)

Both reproduce identically on 1.3.1 and are independent of the ExecuTorch version.

1. **`--all-variants` produces an fp32 program under the `int8` filename** for
   `style_transfer_{candy,mosaic,rain_princess,udnie}` and `efficientnet_v2_s`. Their
   `export.py` calls `lower(ep, config["backend"])` and never reads `variant["quantized"]`,
   so both declared variants come out byte-identical. The *published* artifacts are
   genuinely quantized (`style_transfer_candy` int8 1,835,532 B vs fp32 6,816,524 B;
   `efficientnet_v2_s` int8 22,943,904 B vs fp32 85,714,464 B), so **current `main` cannot
   reproduce the published int8 models at all**. 22 other `export.py` files do handle
   `variant["quantized"]` correctly — these were missed.

2. **The published `all-MiniLM-L6-v2` does not reproduce from `main`.** Published v0.9.0 is
   90,992,000 B; `main` produces 90,391,296 B, and the embeddings differ by up to 0.19
   (max abs, on a unit-normalized 384-d vector). Source/`transformers` drift, not ExecuTorch.
