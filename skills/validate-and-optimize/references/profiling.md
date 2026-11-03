# Profile with the bundled ExecuTorch profiler

The scripts live in [tools/executorch-profiler](../../../tools/executorch-profiler/).
They are standalone diagnostic tools, separate from the export harness.
Use `profile_et.py` for XNNPACK op timing/delegation and
`coreml_compute_plan.py` for CoreML ANE/GPU/CPU assignment.

## Environment

Use a separate profiling environment with an ExecuTorch source checkout/build
compatible with the artifact's exporter/runtime version and custom operators.
Do not replace a model's pinned export environment just to enable tracing.
The build helper requires an active virtualenv and must run **from the
ExecuTorch source checkout**, since it invokes relative build scripts:

```bash
# In a compatible ExecuTorch checkout, with its build dependencies installed:
python -m venv .venv
source .venv/bin/activate
bash /absolute/path/to/executorch-export-recipes/tools/executorch-profiler/build_et.sh
# For CoreML extraction/build support on macOS, use instead:
# bash /absolute/path/to/executorch-export-recipes/tools/executorch-profiler/build_et.sh --coreml
python -m pip install colorama pandas tabulate
```

The helper enables XNNPACK, devtools, event tracing, runner utilities, and
LLM/quantized extensions. CoreML analysis additionally needs `coremltools >= 9.0`
and a supported macOS/Apple runtime; `.pte` extraction needs the CoreML-enabled
ExecuTorch build. Direct `.mlpackage`/`.mlmodelc` analysis skips ET extraction.
Follow the chosen ExecuTorch revision's build requirements if its scripts or
CMake options differ from this snapshot.

`profile_et.py` currently searches for optional LLM/quantized dylibs under
`sys.prefix/lib/python3.12/site-packages`. This repository's model environments
use Python 3.10. For LLMs/custom ops on another Python version, adapt those paths
to the profiling environment and verify registration before interpreting a
missing-op failure; a warning alone does not prove the model needs those libs.

## XNNPACK timing and delegation

From the export-recipes root, using the profiling environment's Python:

```bash
python tools/executorch-profiler/profile_et.py --model exports/model.pte --info
mkdir -p exports/profiles/model-baseline
python tools/executorch-profiler/profile_et.py \
  --model exports/model.pte --model-input exports/inputs.pt \
  --output exports/profiles/model-baseline --n-tests 20
```

Save `inputs.pt` as an ordered tuple/list matching the model's `forward`
signature, e.g. `torch.save((image.contiguous(),), "exports/inputs.pt")`.
Use trusted, saved representative fixtures with the required shapes, dtypes,
token ranges, masks, and state. Without `--model-input`, the script calls its
editable `generate_inputs()` stub and then falls back to random static tensors;
that fallback only maps int64, float32, and bool and is a smoke test.

The runner only loads `forward`; `--info` also requires loading that method.
For models exposing `encode`/`decode` or other methods, adapt the runner or use
a matching native/device runner rather than assuming a `--method` option.
`--n-tests` repeats execution with the same inputs and does not reset caches
or separate warmups. It is not a median/p95 benchmark or an output-parity check.

Always supply a pre-created `--output` directory: the script does not create
it and its empty default resolves the files under `/`. It writes `etdump.etdp`
and `debug.bin`, then prints Inspector's per-op totals, call counts, timing
share, and delegated-call ratios. Use a fresh directory for each run; an existing
dump prompts for reuse and can describe an older artifact/input. Nested method,
delegate, and operator events can overlap; do not sum them as independent work
or treat the call ratio as the fraction of the original graph delegated.

Investigate expensive portable fallbacks and partition boundaries, then confirm
any candidate's quality and performance on the target device. These host timings
do not establish Android/iOS latency.

## CoreML compute-device assignment

```bash
python tools/executorch-profiler/coreml_compute_plan.py \
  --model exports/model-coreml.pte --show-ops
python tools/executorch-profiler/coreml_compute_plan.py \
  --model exports/model-coreml.pte --show-non-ane --compute-units all
# A standalone CoreML model is also accepted:
python tools/executorch-profiler/coreml_compute_plan.py \
  --model /path/to/model.mlpackage --compute-units cpu_and_ne --show-ops
```

The tool extracts CoreML partitions to temporary directories and reports
preferred ANE/GPU/CPU assignment for operations with a compute plan. It accepts
MLProgram `.mlpackage`/`.mlmodelc` models; constants without a plan are omitted.
`--show-ops` groups op types per device; `--show-non-ane` groups those outside ANE.

For a `.pte`, the default comes from its `compute_units` compile specs when
available; otherwise it is `cpu_and_ne`, also the bare-model default. Overrides
are `all`, `cpu_and_ne`, `cpu_and_gpu`, and `cpu_only`. If partitions declare
different settings, the script warns and uses one shared setting (the first
sorted value); this does not reproduce a per-method policy such as Whisper's
encoder/decode split. Analyze those methods/partitions separately under their
actual settings before drawing conclusions.

This is a plan on the **current host**, not measured op latency or proof of iOS
dispatch. An override explores a plan; it does not change the exported `.pte`.
Record host/OS, artifact hash, selected compute units and their source, and
per-partition counts. Confirm correctness and benchmark on the intended Apple
device before changing a recipe's compute units or precision.
