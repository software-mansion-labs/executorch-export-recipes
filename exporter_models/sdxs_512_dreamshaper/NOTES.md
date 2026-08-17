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

## SDXS is extremely quantization-sensitive

The ~14.6x scheduler amplification magnifies any weight error. Measured against a
seed-42 CPU/fp32 reference (mean pixel error on [0, 1]):

| variant | size | mean err | verdict |
| --- | --- | --- | --- |
| xnnpack fp32 | 1.76 GB | 1.8e-5 | ship |
| coreml fp16 | 880 MB | 0.002 | ship |
| 8da4w (linears only) | 1.0 GB | 0.115 | coherent but degraded, dropped |
| int8 PT2E (convs + linears) | 557 MB | 0.335 | pure noise, dropped |

Quantizing the **convolutions** destroys the image; quantizing only **linears** is
coherent-but-degraded. bf16 is also wrong here (weight-rounding probe: bf16 mean
0.00813 / max 0.5531 vs fp16 0.00079 / max 0.0951) — use fp16, never bf16.

## CoreML: compute_unit must exclude the ANE

Exporting with the default `ComputeUnit.ALL` produces a `.pte` that partitions and
loads fine on macOS but whose `denoise` method FAILS TO LOAD on device
(`MILCompilerForANE ... ANECCompile() FAILED`, CoreMLBackend error `0x23`) — the SDXS
UNet is too big for the ANE compiler. `export.py` pins
`compute_unit=ct.ComputeUnit.CPU_AND_GPU`. Delegation with that setting is ~99.7%
(denoise 1212/1216 ops), verified with the executorch-profiler.

`minimum_deployment_target` is `iOS17` here (not the `iOS18` in
`skills/backends/coreml`) — that is what produced the shipped, device-verified `.pte`.

## MLX is published but NOT wired into react-native-executorch

`mlx/sdxs_512_dreamshaper_mlx_int4.pte` exists in the HF repo and this exporter
reproduces it, but `models.ts` ships only XNNPACK fp32 + CoreML fp16. MLX has no
`aten.native_group_norm`, and a diffusion UNet has GroupNorm in every ResBlock, so
`denoise` fragments into **28 subgraphs** (25x group_norm + 2x upsample_nearest2d);
`decode` into 4. On an iPhone 17 Pro that measured ~3x SLOWER than CoreML (though ~4x
faster than XNNPACK) at worse quality. An unquantized fp32 MLX export fragments
identically, proving the split is op coverage and not the 4w dequant pattern — so no
dtype change fixes it. Revisit only if MLX gains `native_group_norm`.

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
