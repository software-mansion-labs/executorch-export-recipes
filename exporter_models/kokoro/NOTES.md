# kokoro onboarding notes

English/standard only (matches the recipe's single-language export). German/Polish
per-language export logic lives upstream in `github.com/NorbertKlockiewicz/kokoro-export`
and is out of scope.

## Status

- **duration_predictor_std.pte** — exports clean; real export verified `io_ok` against
  `config.json` (methods `forward_32/64/128`).
- **synthesizer_coreml_fp32.pte** — exports clean and verified against a real
  af_heart voice plus the published duration predictor's own `d`: spectral
  correlation 0.9993 and RMS ratio 0.9991 against eager. On an iPhone 16 it is
  627 ms warm against 4741 ms for the published XNNPACK build (7.4 s of audio,
  so 11.8x realtime versus 1.6x), after a one-time 12 s compile that is cached
  across launches.
- **synthesizer_std.pte** — the data-dependent guard that used to block re-export
  (`Could not guard on data-dependent expression u7 < 1`) is fixed, see below. The
  xnnpack path has not been re-exported yet, but the blocker is gone.

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

## Why Core ML is synthesizer-only, fp32, CPU_ONLY

- **No duration predictor.** The shipping contract is a single dynamic `forward`
  (see the v0.10.0 artifact). coremltools cannot express it: dynamic `F.pad` on a
  rank > 1 tensor is unsupported, and the `torch.cat` workaround makes
  `torch.export` derive a non-equality constraint on the token dim. A static
  `forward_32/64/128` Core ML build does export and run correctly on device, but
  the runtime feeds the exact token count without padding, so it would only fit
  chunks that land exactly on those sizes.
- **fp32.** In fp16 the decoder lands at 0.973 spectral correlation against eager
  versus 0.9993 in fp32.
- **CPU_ONLY.** `ComputeUnit.ALL` gave a 25 minute compile for one method of the
  duration predictor and an execute failure for another; CPU_ONLY is the 7.6x win
  above.
