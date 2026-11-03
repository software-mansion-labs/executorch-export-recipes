---
name: recover-export
description: Diagnose and recover a failing ExecuTorch export or exported-model runtime failure in this repository, including dependency, checkpoint, tracing, lowering, serialization, and delegate compatibility problems.
---

# Recover an export

Find the earliest failing stage, fix its cause, and validate the resulting
artifact. A successful retry that merely writes a `.pte` is not completion.
Read the model's `NOTES.md`, its dependency pins, and the relevant
[backend skill](../backends/) before changing the recipe.

## Capture a reproducible failure

- Record the repo commit, model, size, backend, variant, exact command, Python
  version, package versions, upstream checkpoint revision, and target runtime
  build. Include the resolved config: local files override fetched HF configs.
- Reproduce one failing variant in its existing locked environment. Use a fresh
  output directory so a stale `.pte` cannot be mistaken for success. Multi-PTE
  models intentionally emit their full set; preserve that contract.
- Retain stderr and the complete traceback when available. The CLI wraps export
  exceptions as `export_failed`; that message alone may hide the original
  exception. Use the same environment to reproduce through the runner and show
  its chained exception, rather than guessing from the final message:

  ```python
  from exporter import registry
  from exporter.pipeline import export_variant

  model, backend, size = "facemesh", "xnnpack", None
  spec = registry.load(model, backend=backend, size=size)
  export_variant(
      spec, spec.default_variant(),
      export_script=str(registry.export_script_path(model, backend)),
      output_dir="exports/debug-facemesh", size=size,
  )
  ```

  Adapt the selection to the actual failure. For `multi_pte` models use
  `export_all` from `exporter.pipeline`; it passes `config["variants"]`.

## Localize the failure

| First failing stage | Evidence to collect | Recovery direction |
| --- | --- | --- |
| Environment or imports | Lock, interpreter, package versions, missing shared library or symbol | Restore the model's verified stack. Inspect the backend-specific project as well as the model root. Fork-pinned Vulkan/MLX builds may require the source-built environment described in the Vulkan skill. Do not upgrade the whole repo to fix one model. |
| Config or checkpoint | Resolved local/HF path, repo/revision, shard index, missing or unexpected state-dict keys | Fix access, the pointer, preprocessing, or checkpoint selection. Verify size and Base/Instruct identity. Do not hide a mismatch with `strict=False` unless the omitted keys are understood and justified. |
| Eager reference | Real inputs, tokenizer/preprocessing, dtype/layout, shapes, masks, cache state | Make the source model run correctly first. A broken reference cannot diagnose the exporter. |
| `torch.export` capture | Earliest unsupported op or data-dependent branch, shape constraints, smallest reproducer | Fix the wrapper or make a semantics-preserving vendored patch. Do not specialize a data-dependent branch merely because one example takes it. Test shape boundaries after changing constraints. |
| Edge conversion or lowering | First failing op/pass, partitioning output, unsupported shapes/dtypes | Isolate the failing pass or subgraph. Use supported decompositions or justified portable fallback, then recheck parity and delegation. Never silently switch the advertised backend. |
| Serialization or runtime load | Artifact path/hash, delegate libraries, op/schema versions, runtime log | Match exporter and runtime builds. MLX schema unions and Vulkan shader registration can fail despite successful export. A missing delegate is not an accuracy problem. |
| Runtime output or performance | Same input bytes in eager/graph/runtime, intermediate differences, timing trace | Follow [validate-and-optimize](../validate-and-optimize/SKILL.md). Check contiguous inputs, preprocessing, caches, and delegate compatibility before changing model math or tolerances. |

## Retry with evidence

Change one suspected cause at a time. Keep the original checkpoint, input
fixtures, logs, and last known good artifact. Record each attempt's hypothesis,
change, and observed result; discard unsupported hypotheses. Preserve the
original model API and published shapes unless a contract change is requested.

Prefer a small fix in the owning model. Mark vendored changes with
`# EXPORT PATCH` and explain the source/revision, symptom, and parity evidence
in its `NOTES.md`. For a runtime defect, capture a minimal reproducer and the
required runtime fix/version; avoid accumulating graph rewrites that mask it.
Do not open upstream issues or publish artifacts unless that is in scope.

Retry a transient download/resource failure when its cause has been addressed.
If the same deterministic error repeats with no new evidence, stop rerunning
the expensive export: reduce it to the failing op/stage or report the concrete
blocker. Missing access, a required device, or an unavailable runtime build
should be reported explicitly, with the next command or prerequisite needed.

## Close the loop

Rerun the original variant in the intended environment, then load the actual
produced file in a compatible runtime. Verify method metadata and constants,
representative output parity, and intended delegation using
[validate-and-optimize](../validate-and-optimize/SKILL.md). Recheck any sibling
variants affected by shared code or dependency changes. Report export, host
validation, device validation, and benchmarking as separate results; mark any
unavailable check as unverified.
