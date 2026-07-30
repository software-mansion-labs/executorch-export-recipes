# kokoro onboarding notes

English/standard only (matches the recipe's single-language export). German/Polish
per-language export logic lives upstream in `github.com/NorbertKlockiewicz/kokoro-export`
and is out of scope.

## Status

- **duration_predictor_std.pte** — exports clean; real export verified `io_ok` against
  `config.json` (methods `forward_32/64/128`).
- **synthesizer_std.pte** — config contract is correct (probed from the published HF
  artifact), but **re-export is blocked under torch 2.12**: `torch.export` raises a
  data-dependent guard error (`Could not guard on data-dependent expression u7 < 1`)
  on a slice deep in the decoder/istftnet graph. The export.py synthesizer path
  faithfully matches the recipe's non-dynamic path (`KModel(... disable_complex=True)`,
  `remove_weight_norms`, `duration_padding=None`), so the recipe-as-vendored hits the
  same wall on this torch version. The published `synthesizer_std.pte` predates this
  stricter data-dependent guard.

## Unblock options for the synthesizer (future)

- Pin an older torch (the recipe's era) where the data-dependent slice guard does not fire.
- Add a minimal `torch._check` / `guard_or_false` shim at the offending slice in the
  vendored decoder forward (real model surgery; trace the symbol with
  `TORCH_LOGS="dynamic"` / tlparse first).
- Export via the recipe's `--dynamic` path (explicit `Dim` shapes for tokens/duration),
  which may resolve the symbolic constraint differently than the static inp-128 path.

The config keeps both variants so the contract is recorded; only the synthesizer's
re-export is deferred.
