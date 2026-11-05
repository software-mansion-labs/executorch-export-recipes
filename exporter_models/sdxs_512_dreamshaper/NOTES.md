# sdxs_512_dreamshaper notes

SDXS-512-DreamShaper text-to-image, exported from `IDKiro/sdxs-512-dreamshaper` as
ONE multi-method `.pte` (`encode` / `denoise` / `decode`) per backend. All shapes are
static, so no `get_dynamic_dims_*` methods are needed. `_pipeline.py` owns the three
wrappers and the example inputs; each `<backend>/export.py` differs only in lowering.

## What is deliberately NOT exported

- **The scheduler.** SDXS is a distilled ONE-step, no-CFG pipeline
  (`num_inference_steps=1`, `guidance_scale=0`, single timestep 999). At that single
  step the DEIS update is *exactly* linear in (latents, eps):
  `clean = 14.642591 * latents + (-14.579279) * eps` (lstsq residual 3.8e-6,
  seed-independent). Those pinned coefficients live in the react-native-executorch
  task; re-deriving or looping DEIS on the runtime side is wrong.
- **The tokenizer.** Shipped as `tokenizer.json` in the HF repo, not a `.pte`.
- **A `get_model_schema` constant method.** Every dimension here is static, so
  react-native-executorch populates the exported spec from ExecuTorch `MethodMeta`
  alone; the companion schema is only needed for dynamic/enum dims or runtime
  constraints. The shipped `.pte` carries no such method either.
- **The TAESD unscale.** `decode` bakes `image = clamp(raw / 2 + 0.5, 0, 1)` into the
  graph and returns RGB in [0, 1]; the VAE `scaling_factor` is 1.0 (no 0.18215).

## SDXS is quantization-sensitive: static int8 breaks it, dynamic int8 does not

The ~14.6x scheduler amplification magnifies any weight error. Measured against a
seed-42 CPU/fp32 reference (mean pixel error on [0, 1]):

| variant | size | mean err | verdict |
| --- | --- | --- | --- |
| xnnpack fp32 | 1.76 GB | 1.8e-5 | ship |
| coreml fp16 | 880 MB | 0.002 | ship |
| 8da4w (linears only) | 1.0 GB | 0.115 | coherent but degraded, dropped |
| int8 PT2E static (convs + linears) | 557 MB | 0.335 | pure noise, dropped |

`sdxs_512_dreamshaper_xnnpack_int8.pte` (813 MB) instead uses PT2E **dynamic** int8
activations with int8 per-channel weights on every conv and Linear of `denoise` and
`decode`; `encode` stays fp32. 16 images (8 prompts x 2 seeds) against fp32: mean pixel
error 0.075, same scenes, CLIP score 34.94 vs 34.20. Galaxy S26 Ultra per image:
1323 -> 845 ms (denoise 922 -> 514, decode 373 -> 303). Lowering needs
`XnnpackDynamicallyQuantizedPartitioner` ahead of `XnnpackPartitioner`; one partitioner
over mixed nodes fails with "Invalid partition, found dependency cycles".

bf16 is also wrong here (weight-rounding probe: bf16 mean 0.00813 / max 0.5531 vs fp16
0.00079 / max 0.0951) — use fp16, never bf16.

## CoreML: compute_unit must exclude the ANE

Exporting with the default `ComputeUnit.ALL` produces a `.pte` that partitions and
loads fine on macOS but whose `denoise` method FAILS TO LOAD on device
(`MILCompilerForANE ... ANECCompile() FAILED`, CoreMLBackend error `0x23`) — the SDXS
UNet is too big for the ANE compiler. `export.py` pins
`compute_unit=ct.ComputeUnit.CPU_AND_GPU`. Delegation with that setting is ~99.7%
(denoise 1212/1216 ops), verified with the executorch-profiler.

`minimum_deployment_target` is `iOS17` here (not the `iOS18` in
`skills/backends/coreml`) — that is what produced the shipped, device-verified `.pte`.

## MLX is NOT published (removed 2026-08-21)

`mlx/sdxs_512_dreamshaper_mlx_int4.pte` was published for a while and has been
removed from the HF repo, which now carries `mlx/NOTES.md` in its place. This
exporter is kept so the decision can be re-tested against a future
ExecuTorch/MLX release; `model.json` still lists `mlx` and `mlx/config.json` is
still tracked here for that reason.

MLX loses on all three axes:

| | xnnpack fp32 | coreml fp16 | mlx int4 |
| --- | --- | --- | --- |
| size | 1760.6 MB | **880.7 MB** | 1034.8 MB |
| latency | slowest | **baseline** | ~3x slower than CoreML |
| mean pixel err | 1.8e-5 | **0.002** | degraded, not quantified |

Being *larger* than CoreML fp16 while quantized to 4 bits is what settles it:
the size win that justifies shipping MLX for CLIP and distiluse is absent here.
Latency was measured on an iPhone 17 Pro (~4x faster than XNNPACK, but XNNPACK
is not the iOS build we ship). On quality, the MLX build is 4w weight-only on
linears with convs and embeddings left fp32, the same class as the `8da4w` row
in the table above that measured 0.115 and was rejected as coherent but
degraded; MLX was observed worse in the original comparison but was never
separately measured against the seed-42 reference.

**Root cause is op coverage.** The MLX backend registers no
`aten.native_group_norm` and no `aten.upsample_nearest2d` (confirmed absent from
`backends/mlx/ops.py` on upstream `main`, which does register `layer_norm`,
`native_layer_norm`, `rms_norm` and `_native_batch_norm_legit_no_training`). A
diffusion UNet has GroupNorm in every ResBlock, so `denoise` fragments into
**28 subgraphs** (25x group_norm + 2x upsample_nearest2d) and `decode` into 4.
`encode` is the control: pure transformer, LayerNorm only, delegates in one
piece. An unquantized fp32 MLX export fragments identically, proving the split
is op coverage and not the 4w dequant pattern, so no dtype change fixes it.

Filed upstream as https://github.com/pytorch/executorch/issues/22017. Revisit
only if the MLX backend gains `native_group_norm`.

As with the other MLX models, `mlx/export.py` needs executorch built from the
software-mansion-labs fork (version-locked to the runtime MLX backend); a pip
ExecuTorch partitions and writes a `.pte` that fails at execute with
`Error::InvalidProgram`. See `mlx/pyproject.toml`.

## Stack

`diffusers` is pulled in with the same pin as bk_sdm_tiny. The original export ran on
`transformers` 5.0rc1; this project pins 5.14.1 to match the rest of the repo.

## Re-export verification (xnnpack fp32, this harness)

- **Contract** — the produced `.pte` exposes `encode` / `denoise` / `decode` and its
  `MethodMeta` I/O matches the published `xnnpack/config.json` exactly.
- **Eager parity** — all three methods are bit-exact (maxabs `0.0`) against the
  seed-42 reference captured from the original CPU/fp32 pipeline, so the
  `transformers` 5.0rc1 -> 5.14.1 move does not move numerics.
- **Graph** — `decode` and `denoise` are structurally identical to the shipped
  artifact (same operator, delegate, instruction and value counts). `encode` has ONE
  extra `aten::add.out` outside the delegate (269 vs 268 instructions, same 49
  XNNPACK delegate calls) — a `transformers` fusion difference, numerically inert
  per the parity check above.
- File size 1 760 575 372 B vs the shipped 1 760 575 244 B.
