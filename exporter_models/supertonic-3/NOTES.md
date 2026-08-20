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

`coreml` is therefore **not** in this model's `model.json` backends. The exporter is
kept so the finding can be re-tested against a future coremltools. Related:
apple/coremltools#2825 and `docs/coremltools-2825-repro.md`.
