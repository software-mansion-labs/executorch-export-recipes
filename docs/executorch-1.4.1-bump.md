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

1. **`--all-variants` produced an fp32 program under the `int8` filename** for
   `style_transfer_{candy,mosaic,rain_princess,udnie}` and `efficientnet_v2_s` — FIXED in
   commit 916398b. Their `export.py` called `lower(ep, config["backend"])` and never read
   `variant["quantized"]`, so both declared variants came out byte-identical, while the
   *published* artifacts are genuinely quantized. No commit of those files had ever
   contained the string "quantized": the step was never carried over when 171c4fd reshaped
   these models onto the per-model `export()` contract, though 22 other export scripts did
   get it. Nothing caught it because the harness verifies `method_meta` I/O shapes, and an
   int8 XNNPACK model has exactly the same shapes and dtypes as its fp32 twin — only file
   size differs. Verified after the fix on `style_transfer_candy`:

   | file | ours | published |
   |---|---|---|
   | `..._xnnpack_fp32.pte` | 6,818,060 | 6,816,524 |
   | `..._xnnpack_int8.pte` | 1,836,812 | 1,835,532 |

   Before the fix the int8 was 6,818,060 B, byte-identical to fp32.

2. ~~The published `all-MiniLM-L6-v2` does not reproduce from `main`.~~ **RETRACTED.** That
   was a comparison against the **v0.9.0** tag. The library requests **v0.10.0**
   (`NEXT_VERSION_TAG`), and against v0.10.0 `main` reproduces the published artifact
   exactly (90,391,296 B both sides). There is no reproducibility drift; always check which
   tag the code actually points at before concluding an artifact does not reproduce.

## What the nlp re-exports actually changed

Sizes of the 1.4.1 re-exports vs published **v0.10.0** (13 artifacts, 8 models). Every
`.pte` was confirmed loading and running on a Galaxy S26 Ultra.

- **7 identical in size**: all_minilm_l6_v2, all_mpnet_base_v2, clip_vit_base_patch32_text,
  multi_qa_minilm_l6_cos_v1, multi_qa_mpnet_base_dot_v1 (all xnnpack fp32), plus the
  distiluse and paraphrase xnnpack **8da4w** variants.
- **2 smaller by exactly 128 B**: distiluse and paraphrase xnnpack **fp32**.
- **4 larger by 150-370 KB**: distiluse and paraphrase **coreml** fp16 and fp32.

Not chased down to a root cause (the artifacts work, so it was not worth the compute).
Two things to know if you pick this up: distiluse and paraphrase are the only two nlp
models whose `torchao` pin moved from the nightly to the 0.18.0 release, which makes that
pin the obvious suspect — but their *quantized* 8da4w files match while their *fp32* files
differ, which is the opposite of what a quantization-library change would predict. And
size equality is not byte equality: only `all_minilm_l6_v2` has a true controlled
1.3.1-vs-1.4.1 SHA-256 comparison (identical).
