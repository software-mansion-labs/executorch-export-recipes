# kokoro onboarding notes

English/standard only (matches the recipe's single-language export). German/Polish
per-language export logic lives upstream in `github.com/NorbertKlockiewicz/kokoro-export`
and is out of scope.

## Status

- **duration_predictor_std.pte** / **synthesizer_std.pte** (xnnpack) — unchanged,
  still the default. The data-dependent guard that used to block re-exporting the
  synthesizer (`Could not guard on data-dependent expression u7 < 1`) is fixed,
  see below.
- **duration_predictor_std_coreml_fp32.pte** — single `forward`, token axis fixed
  at 128 (the runtime pads up to it), plus a companion `get_model_schema`.
- **synthesizer_std_coreml_fp32.pte** — token axis fixed at 128, **frame axis
  dynamic (16..296)**, so audio length follows the predicted durations exactly and
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
