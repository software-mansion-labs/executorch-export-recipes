# kokoro onboarding notes

The Core ML export covers all three language variants (standard, polish, german).
The xnnpack and mlx exports still cover the standard weights only and raise on the
other two; their published Polish/German `.pte` were produced upstream in
`github.com/NorbertKlockiewicz/kokoro-export`.

Language variants ride the harness's `{size}` axis: `sizes: [standard, polish,
german]` with `config: "{backend}/{size}/config.json"`, one `export.py` per
backend picking the checkpoint from `config["size"]`. The Polish and German
fine-tunes live at `finetunes/kokoro_{polish,german}_converted.pth` in the
published HF repo; they share the base architecture and config exactly (same
five modules, same per-module parameter counts), so only the weights differ.

## Status

- **duration_predictor_std.pte** / **synthesizer_std.pte** (xnnpack) — unchanged,
  still the default. The data-dependent guard that used to block re-exporting the
  synthesizer (`Could not guard on data-dependent expression u7 < 1`) is fixed,
  see below.
- **duration_predictor_{std,pl,de}_coreml_fp32.pte** — single `forward`, token
  axis fixed at 128 (the runtime pads up to it).
- **synthesizer_{std,pl,de}_coreml_fp32.pte** — token axis fixed at 128, **frame
  axis dynamic (16..296)**, declared through a companion `get_model_schema`, so audio length follows the predicted durations exactly and
  no tempo scaling is needed. Verified against the exact exported graph at
  D = 296/220/120/60: spectral correlation 0.99981 / 0.99987 / 0.99998 / 0.99993.
  On an iPhone 16 the static-frame build ran 7.4 s of audio in 627 ms warm against
  4741 ms for published XNNPACK; the dynamic build costs about 5% on top of that
  at full length and drops to ~180 ms at D = 120.

## What was fixed in `_kokoro_model/istftnet.py`

Two changes in `SineGen`, both verified against the original formulation:

1. `rad_values = (f0_values / sampling_rate) % 1` is written as `x - floor(x)`.
   coremltools converts `% 1` to a tensor of **zeros** (apple/coremltools#2836,
   divisor 1 only), which silently collapses the excitation. No error is raised,
   the audio is just wrong.
2. The pulse-train branch used `tmp_cumsum[idx, u_loc[idx, :, 0], :]` followed by
   `temp_sum[1:] -= temp_sum[:-1]`. The boolean-mask select produces an unbacked
   size and ET's view -> view_copy pass then fails to guard on it, which is what
   blocked re-export. `_segmented_wrapped_phase` computes the same phase as a
   segmented prefix sum wrapped mod 1 (cos differs by <5e-6 in fp32), with static
   shapes only. Wrapping also keeps every intermediate in [0, 1) instead of
   growing to hundreds of cycles.

`aten.cummax` is not in the core ATen opset, so the scan is built from
`maximum`/`cat`/`slice` and unrolls to ceil(log2(T)) steps. Integer literals
assigned into float tensors (`rand_ini[:, 0] = 0`) must be float literals, or the
Core ML converter rejects the `slice_update` on a dtype mismatch.

## Why the token axis is static, and why fp32 / CPU_ONLY

- **Static tokens.** coremltools cannot express that axis: dynamic `F.pad` on a
  rank > 1 tensor is unsupported, and the `torch.cat` workaround makes
  `torch.export` derive a non-equality constraint on the token dim. Both models
  therefore fix it at 128 and the runtime pads up to it, which is what the
  shipping XNNPACK duration predictor already does internally (its published
  .pte contains the same `constant_pad_nd`).
- **Dynamic frames.** A static frame axis is not shippable: any frame padding
  destroys the audio (-1.7 to -4.8 dB against a 222 dB control), because AdaIN
  normalizes over time. Four coremltools limitations had to be worked around to
  keep the axis dynamic; they are documented in `coreml/export.py` and filed as
  apple/coremltools#2837 and #2838.
- **fp32.** In fp16 the decoder lands at 0.973 spectral correlation against eager
  versus 0.9993 in fp32.
- **CPU_ONLY.** `ComputeUnit.ALL` gave a 25 minute compile for one method of the
  duration predictor and an execute failure for another; CPU_ONLY is the 7.6x win
  above.

## Measuring this model

Sample-wise SNR is the wrong metric here. The excitation draws its initial phase
from a runtime `aten::rand`, so two runs of the *same* graph on the same input
correlate only ~0.88 sample-wise across phase draws. Compare magnitude spectra
instead: same-phase spectral correlation floors at ~0.999, which is what the
numbers above report. Compare against the exact exported graph, not against a
differently padded eager model.

## Verifying a language variant

`.pte` versus the exact exported eager graph, per language voice, at several
frame counts (spectral correlation):

| D | standard | polish | german |
|---|---|---|---|
| 296 | 0.999811 | 0.999968 | 0.999941 |
| 220 | 0.999873 | 0.999976 | 0.999955 |
| 120 | 0.999979 | 0.999984 | 0.999946 |
| 60  | 0.999934 | 0.999988 | 0.997674 |
| 16  | -        | 0.920152 | 0.976817 |

Always measure the **eager-vs-eager floor at the same D** before reading a
number as a regression. The excitation draws a random initial phase, and short
outputs have too little spectral resolution to see past it: at D=16 two runs of
the same graph correlate only ~0.93 (polish) / ~0.96 (german), and at D=60 the
german floor ranges 0.9962..0.9991 over six draws. Every number above is at or
above its own floor.

Building inputs for the check: use the real predicted durations over the longest
token prefix that fits in D frames, topped up to hit D exactly. Rescaling all 128
tokens into D < 128 frames needs per-token counts below 1 and silently produces
garbage indices.

## Vulkan: evaluated, not viable yet

No `vulkan/export.py` is committed. Both modules were lowered and the duration
predictor was run on a Galaxy S26 Ultra (Adreno 840), but the result does not
justify an exporter yet. This section records the recipe and the numbers so the
work is not repeated from scratch.

### What it takes to lower at all

Four things, in the order they bite:

1. **`attn_implementation = "eager"` on the PLBERT.** SDPA builds its additive
   mask with `-inf`, and the Vulkan serializer writes floats through flatc's
   JSON, which has no spelling for a non-finite value, so the export dies with
   `cannot parse value starting with: -`. Setting it post-hoc works:

   ```python
   model.bert.config._attn_implementation = "eager"
   for m in model.bert.modules():
       if hasattr(m, "config"):
           m.config._attn_implementation = "eager"
   ```

   It is also worth a lot on its own: the predictor goes from 25 Vulkan
   subgraphs to 2. Same finding as `all_minilm_l6_v2`.
2. **`aten.constant_pad_nd` blocklisted.** `_padded_lstm` pads by
   `target - seq_len`, which on a dynamic axis is a SymInt, so the AOT emits a
   ValueList where Vulkan's `Pad.cpp` calls `get_int_list` and requires a
   literal IntList. The runtime aborts at prepack with `Expected value to have
   type IntList, got VALUELIST`. Unlike `Split.cpp`, `View.cpp` and
   `Expand.cpp`, which all use `extract_int_or_symint_list`, `Pad.cpp` was
   missed. A real fix is more than swapping the extractor, because the pad
   offsets are baked into a params buffer at build time and would have to be
   refreshed on resize, so these four pads stay on the host.
3. **pytorch/executorch#22504** — `AdaLayerNorm` calls `F.layer_norm` with no
   affine parameters, and Vulkan threw on a missing weight. PR #22506.
4. **pytorch/executorch#22505** — the int32 `aten.eq` shader is generated under
   a name the dispatcher never asks for. PR #22507.

### Why it is not viable yet

The duration predictor lowers cleanly and is **slower than XNNPACK**: 98.0 ms
against 79.7 ms per execution at the 128-token bound, 20 executions each on a
cooled device. It is dominated by the 128-step unrolled LSTM, which is 17750
ops in a single delegate, and a sequential chain of tiny matmuls is the worst
case for a GPU.

The synthesizer lowers but **fragments into 75 delegates**, and the host
staging that implies is 218 MB of planned buffer (against 0.7 MB for the
predictor's 6 delegates). Running it at the upper bound rebooted the device
twice. Check `non_const_buffer_sizes` and the delegate count on any future
attempt BEFORE running it.

The fragmentation is not intrinsic. It comes from ops Vulkan does not
implement:

| cause | splits |
| --- | --- |
| `aten.var.correction` (AdaIN's instance norm) | 70 |
| `aten.reciprocal` | 48 |
| `aten.floor` (the phase wrap in `_segmented_wrapped_phase`) | 19 |
| no valid representation, i.e. texture limits on `120*frames+1` shapes | 25 |
| `aten.convolution` rejected | 7 |

The first three are trivial elementwise and reduction ops and account for 137
of the 160 unimplemented-op splits. The texture-limit group needs buffer
storage for audio-length 1D tensors. Adding those three ops is the prerequisite
for a second attempt; qwen3 went from 57 delegates to 1 and from a loss to a
2.7x prefill win once its one split cause was removed, so the same collapse is
plausible here, but it is unmeasured.

## AdaIN is 15% of the synthesizer, and `var_mean` does not fix it

Profiling the published XNNPACK synthesizer with etdump put `mean` + `var`
inside `CustomInstanceNorm1d` at 172 ms of 1105 ms, which is 77% of all
non-delegated time. Replacing the two reductions with a single
`torch.var_mean(x, dim=[2], correction=0, keepdim=True)` is the obvious fix and
**does not export**:

```
Operator torch._ops.aten.var_mean.correction is not in Core ATen opset
```

`to_edge` refuses it outright. Getting past that needs
`EdgeCompileConfig(_core_aten_ops_exception_list=[torch.ops.aten.var_mean.correction])`
AND a `var_mean.correction_out` kernel linked into the RNE runtime, which is
unverified. The change is not in the tree; `CustomInstanceNorm1d` still uses
separate `mean` and `var`.
