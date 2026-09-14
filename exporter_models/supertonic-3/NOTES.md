# supertonic-3 notes

## Core ML: not supported (export fails)

`coreml/export.py` exists (all four sub-models) but Core ML **cannot export this
model**. Tested 2026-08-20 by driving the exporter from the prebuilt fork
environment, which bypasses the earlier `uv` blocker (this model pins `executorch`
to the software-mansion-labs fork, and uv tries and fails to build it from source):

```
PYTHONPATH=<repo>/src /path/to/.venv-aot/bin/python -m exporter.cli \
    export supertonic-3 --backend coreml --all-variants --output-dir <out>
```

It fails on the first sub-model (`duration_predictor`) during conversion:

```
Placeholder val must be a tensor or fake tensor, but got type <class 'torch.SymInt'>, value s16 + 1
```

That is the dynamic-shape limitation hitting at **export** time rather than at
execute. Every sub-model here carries dynamic dims (`T` for text 8..256, `L` for
latent 4..1000), and a symbolic dim reaches the Core ML converter as a placeholder
it cannot represent. `vocoder`, which takes only a float latent and was the best
candidate, is never reached because the run aborts on the first failure.

No Core ML `.pte` is published for this model. The exporter and its
`coreml/config.json` are kept, and `coreml` stays in `model.json` backends, so the
command above actually runs and the finding can be re-tested against a future
coremltools. (It was previously dropped from that list, which made the command
above fail with a SpecError before it ever reached coremltools.) Related:
apple/coremltools#2825 and `docs/coremltools-2825-repro.md`.

## Vulkan: shipped (fp16, Android)

`vulkan/export.py` reuses the XNNPACK model plumbing and swaps in
`VulkanPartitioner(force_fp16=True)`. On a Galaxy S26 Ultra (Adreno 840),
median ms per execution at T=512 / L=1000:

| sub-model | vulkan fp16 | xnnpack fp32 | speedup |
| --- | --- | --- | --- |
| duration_predictor | 18.7 | 33.2 | 1.78x |
| text_encoder | 72.3 | 140.9 | 1.95x |
| vector_estimator | 682.5 | 1253.7 | 1.84x |
| vocoder | 865.8 | 1284.3 | 1.48x |

2.12x end to end at the default 8 flow-matching steps, where `vector_estimator`
is 84% of the total. Outputs match the fp32 CPU references at cosine 1.000000,
0.999414, 0.999994 and 0.999977.

Getting there needed four ExecuTorch fixes, all upstreamed and cherry-picked
into the software-mansion-labs fork:

- pytorch/executorch#22399 - do not partition ops whose symbolic arguments the
  runtime reads at build time (symbolic `clamp` bound, symbolic `constant_pad_nd`,
  non-4d batch norm).
- pytorch/executorch#22401 - size the texture-vs-buffer choice by the bound, not
  the trace hint. Without it `text_encoder` silently computed garbage
  (cosine 0.41) whenever it ran longer than the traced length.
- pytorch/executorch#22402 - only rewrite `addmm`/`mm` to `linear` when the
  weight is a constant. `duration_predictor` multiplies by a runtime-derived
  matrix, which became a `linear` whose weight the runtime then tried to prepack.
- pytorch/executorch#22403 and #22406 - keep gathers on the GPU. `x[:, :, idx]`
  was unsupported, and each occurrence split the delegate, so `vector_estimator`
  lowered to 29 delegate calls. Now 2, worth 11-17% end to end.

### T bound: 512, not 256

The published XNNPACK `.pte` files declare `T` up to **512**; `schema.json` and
`_T_MAX` in this repo said 256, so the repo had drifted from what is live. Both
were moved to 512 so the two backends agree with production. Re-exporting
XNNPACK from this repo now reproduces the shipped bound.

### Where the remaining time goes

`ETVK_DISPATCH_TRACE=<path>` (env-gated, no rebuild needed) on one
`vector_estimator` execution shows **1246 dispatches, only 56 of them
convolutions** - those 56 are ~278 ms of the 682. The rest is data movement:
202 `binary_mul`, 166 `view_texture`, 158 `permute_texture3d`, 162 staging
conversions, 36 layer norms.

**Next lever is the shared GEMM kernel, not this model.** Measured head to head
on identical arithmetic ([2,512,1000], 512->2048->512, 4.19 GMAC), `conv1d_pw`
takes 9.94 ms and `aten.linear` 10.70 ms, both about **840 GFLOP/s**, which is
roughly 15-20% of the device's fp16 peak. Neither used cooperative matrix
instructions. So there is no win in respelling pointwise conv1d as a matmul
(that was measured and rejected); the multiple that is left lives in the tiled
GEMM shader itself and would benefit every Vulkan model, not just this one.
