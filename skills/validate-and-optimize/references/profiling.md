# Profile with the bundled ExecuTorch profiler

The scripts live in [tools/executorch-profiler](../../../tools/executorch-profiler/).
They are standalone diagnostic tools, separate from the export harness.
Use `profile_et.py` for XNNPACK op timing/delegation and
`coreml_compute_plan.py` for CoreML ANE/GPU/CPU assignment.
For Vulkan, use the source-built Android runner and ETDump workflow below.

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

`profile_et.py` discovers optional LLM/quantized operator libraries relative to
the imported `executorch` package, using `.dylib`, `.so`, or `.dll` filenames.
It does not assume a Python version or virtualenv layout. A missing-library
warning alone does not prove the model needs those operators. Unloadable optional
libraries also produce a warning; models requiring them still need matching
libraries rebuilt with the profiling runtime.

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

Use a separate `--output` directory for each profile; the script creates it
and defaults to the current directory. Paths support `~`. It writes `etdump.etdp`
and `debug.bin`, then prints Inspector's per-op totals, call counts, timing
share, and delegated-call ratios. Use a fresh directory for each run; an existing
dump prompts for reuse and can describe an older artifact/input. Reuse preserves
the matching `debug.bin`. Nested method,
delegate, and operator events can overlap; do not sum them as independent work
or treat the call ratio as the fraction of the original graph delegated.

Investigate expensive portable fallbacks and partition boundaries, then confirm
any candidate's quality and performance on the target device. These host timings
do not establish Android/iOS latency.

## Vulkan on-device delegation and timing

Build the Android runtime from the compatible
[ExecuTorch fork](https://github.com/software-mansion-labs/executorch) revision
used by the export/shipping runtime. Enable **`EXECUTORCH_ENABLE_EVENT_TRACER`**
when rebuilding both the runtime and Vulkan backend. This defines
`ET_EVENT_TRACER_ENABLED`; Vulkan then records GPU shader timestamps, not just
CPU dispatch duration. See the
[upstream Vulkan profiling tutorial](https://github.com/pytorch/executorch/blob/main/docs/source/backends/vulkan/tutorials/etvk-profiling-tutorial.md)
and [ETDump build requirements](https://github.com/pytorch/executorch/blob/main/docs/source/etdump.md).

From that ExecuTorch checkout, with its build prerequisites installed and
`ANDROID_NDK` pointing to the NDK, configure a separate profiling build:

```bash
cmake -S . -B cmake-out-android-profile --preset android-arm64-v8a \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON
cmake --build cmake-out-android-profile --target executor_runner -j8
```

Retain the pinned revision's other required build options, custom operators,
and CPU delegates for mixed-backend models. Check `CMakeCache.txt` for the
profiling options and deploy the rebuilt runner and any shared libraries it
needs together. Re-exporting alone does not enable runtime tracing.

```bash
adb shell mkdir -p /data/local/tmp/etvk
adb push cmake-out-android-profile/executor_runner /data/local/tmp/etvk/
adb push /absolute/path/to/model.pte /data/local/tmp/etvk/model.pte
adb shell chmod +x /data/local/tmp/etvk/executor_runner
adb shell /data/local/tmp/etvk/executor_runner \
  --model_path /data/local/tmp/etvk/model.pte \
  --num_executions=20 \
  --etdump_path /data/local/tmp/etvk/model.etdp
mkdir -p /absolute/path/to/executorch-export-recipes/exports/profiles/vulkan
adb pull /data/local/tmp/etvk/model.etdp \
  /absolute/path/to/executorch-export-recipes/exports/profiles/vulkan/model.etdp
python devtools/inspector/inspector_cli.py \
  --etdump_path /absolute/path/to/executorch-export-recipes/exports/profiles/vulkan/model.etdp
```

This command is a trace smoke test with runner-generated inputs. For useful
measurements, supply representative fixtures through the pinned runner's input
options (check `--help`) or a model-specific native runner. A custom runner must
link `etdump`, pass an `ETDumpGen` tracer to the loaded method/module, and write
the collected buffer. Build flags alone do not save a trace.

Inspector exposes delegated shader events (`is_delegated_op`) with timing
statistics alongside portable operator events. Find expensive shaders, CPU
fallbacks, and staging/copy overhead. Vulkan can fuse source ops or dispatch
several shaders per op, so a shader-event ratio is not source-op delegation
coverage. Use the lowered graph and, when available, a matching export ETRecord
to map back to source ops; do not sum nested aggregate and shader events.

If GPU events are absent, check the deployed build, tracer attachment, and GPU
timestamp support before concluding that nothing delegated. Record runtime
commit, artifact hash, device/GPU, inputs, and iteration settings. Separate
warmup from measured inference and remeasure performance without tracing to
quantify its overhead. MoltenVK results do not establish Android performance.

## CoreML compute-device assignment

From the export-recipes repository root, using the profiling environment:

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
