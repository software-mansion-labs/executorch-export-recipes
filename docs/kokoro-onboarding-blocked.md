# kokoro onboarding — BLOCKED (deferred)

kokoro (TTS) was NOT onboarded in the multi-variant wave. It is blocked for
reasons that require either upstream artifacts we don't have locally or a
design decision. Documented here so the next attempt starts from facts, not
re-discovery.

## What's on HuggingFace (`software-mansion/react-native-executorch-kokoro`)

- root `config.json` is the `{modelName}` stub only — **no canonical spec config**
  anywhere (`xnnpack/config.json`, `xnnpack/standard/config.json` → 404).
- Published `.pte` artifacts, per language family:
  - `xnnpack/{german,polish,standard}/duration_predictor_{de,pl,std}.pte`
  - `xnnpack/{german,polish,standard}/synthesizer_{de,pl,std}.pte`
- Plus non-pte assets: `phonemizer/<lang>/*.pte` + `lexicon.json`/`tags.json`,
  and `voices/*.bin` (per-voice style vectors).

So the model surface = **2 component types (duration_predictor, synthesizer)
× 3 language families (de/pl/std) = 6 model PTEs**, plus phonemizer PTEs and
voice bins as assets.

## What the local recipe (`exports/kokoro/`) can produce

- `export_duration_predictor.py` and `export_synthesizer.py` are the two
  "main" exporters (per README; `export_f0n_predictor.py`/`export_encoder.py`/
  `export_decoder.py` are legacy, already bundled into the synthesizer).
- They build `KModel(repo_id="hexgrad/Kokoro-82M", ...)` — downloads the 82M
  base model from HF at export time — and depend on the vendored `model/`
  package (custom_stft, istftnet, modules, postprocessing, stable_adain,
  temporal_scaling) + the `kokoro` pip package (NOT installed in any local
  env) + `misaki`.
- Their CLI args are `--input-size {inp-32,inp-64,inp-128}`, `--max-tokens`,
  `--max-duration`, `--dynamic`, `--bundled` — i.e. **token-count sizing,
  not language**.

## Why it's blocked

1. **No language dimension in the local recipe.** The published artifacts
   differ by *language* (de/pl/std), but the vendored export scripts have no
   language parameter — they size by token count. The README says "Full
   content (including some extra scripts) is available within
   https://github.com/NorbertKlockiewicz/kokoro-export" — the per-language
   export logic lives in that upstream repo, NOT in `exports/kokoro/`. We
   cannot reproduce `duration_predictor_de.pte` etc. from the local recipe
   without that missing piece.
2. **No spec config to verify against.** There's no HF `config.json` with
   I/O shapes, and the I/O shapes can't be derived without running the heavy
   pipeline (which itself can't target the published per-language artifacts).
   A hand-authored config would be fabricated guesswork.
3. **Heavy uninstalled deps.** `kokoro` is absent; a per-family venv from
   `exports/kokoro/uv.lock` (106 deps) would be needed even to run the
   recipe — and that still wouldn't close gap (1).

## What would unblock it (for the next attempt)

- The per-language export logic from `github.com/NorbertKlockiewicz/kokoro-export`
  (how `_de`/`_pl`/`_std` artifacts are produced — likely language-specific
  vocab/voice/phonemizer wiring), OR
- A confirmed mapping `(component, language) → exact recipe invocation`, OR
- A decision to onboard kokoro as an **asset bundle** (ship the published
  PTEs + phonemizers + voices directly with a hand-authored multi-variant
  config whose shapes are probed from the published PTEs via `read_pte_io`),
  rather than re-exporting from the recipe. This last option fits the harness
  if the goal is "re-publish/verify" rather than "re-export from source".

The other two multi-pte models in this wave (bk_sdm_tiny, moonshine_tiny) were
onboarded; kokoro is the one genuine deferral.
