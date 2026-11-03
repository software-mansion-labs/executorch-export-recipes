# Executorch Profiler

For this repository's environment setup, commands, saved-input format, and
known limitations, read the
[profiler guide](../../skills/validate-and-optimize/references/profiling.md).
Run `build_et.sh` from a compatible ExecuTorch source checkout with an active
virtualenv. Always give `profile_et.py` a pre-created `--output` directory.

This repository contains a Python script for profiling Executorch models running on the XNNPACK backend, along with a helper script for building Executorch with event tracing enabled.

It also contains `coreml_compute_plan.py`, which reports how a CoreML-delegated model is dispatched across the Apple Neural Engine (ANE), GPU and CPU.


## Install Executorch (manual setup)

1. Clone the Executorch repository:

   ```bash
   git clone https://github.com/pytorch/executorch
   ```

2. Enter the repository directory:

   ```bash
   cd executorch
   ```

3. Create and activate a virtual Python environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Build and install Executorch with XNNPACK and event tracer enabled:

   ```bash
   CMAKE_ARGS="-DEXECUTORCH_BUILD_XNNPACK=ON \
   -DEXECUTORCH_BUILD_COREML=OFF \
   -DEXECUTORCH_BUILD_DEVTOOLS=ON \
   -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
   -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
   -DCMAKE_BUILD_TYPE=Release" \
   ./install_executorch.sh
   ```

5. Install missing Python packages (at minimum `colorama`):

   ```bash
   pip install colorama
   ```


## Install Executorch (using build_et.sh)

As an alternative to the manual steps above, you can use the `build_et.sh` script in this directory.  
It configures the same CMake options for Executorch (XNNPACK backend and event tracer enabled) and installs required Python packages such as `colorama`.

Pass `--coreml` to additionally build the CoreML backend, which is required to analyze CoreML-delegated models:

```bash
./build_et.sh --coreml
```


## Profiling script usage

Once Executorch is installed and your virtual environment is active, you can run the profiling script, for example:

```bash
python profile_et.py --model=exported_model.pte --output=.
```


## Defining custom input generator

The profiling script expects a user-defined input generator function for your model.  
You should provide it in the script as shown below:

```python
# User-defined input generator
def generate_inputs():
    # <---------------------->
    # DEFINE MODEL INPUTS HERE
    # <---------------------->
    raise NotImplementedError("")
```

Implement `generate_inputs` to return inputs in the format expected by your Executorch model. You can also provide the input as .pt file with *--model-input* argument.

**WARNING**: If not explicitely defined, the profiling script will try to generate random inputs based on model's metadata.


## CoreML compute plan usage

`profile_et.py` measures op timing on the XNNPACK backend. For CoreML-delegated models the more useful question is usually *which compute device each op lands on*: the CoreML runtime decides at compile/load time whether each op runs on the Apple Neural Engine (ANE), the GPU or the CPU. `coreml_compute_plan.py` reports that breakdown so you can see why a model isn't running fully on the ANE — no Swift required.

Requirements:

- `coremltools >= 9.0` (the `MLComputePlan` API the script relies on)
- An Executorch build with the CoreML backend enabled (`./build_et.sh --coreml`)

The script accepts a CoreML-backed `.pte` (every CoreML partition is extracted and analyzed) or a CoreML model directly (`.mlpackage` / `.mlmodelc`):

```bash
# Analyze every CoreML partition embedded in an ExecuTorch program.
python coreml_compute_plan.py --model=exported_model.pte

# Analyze a CoreML model directly.
python coreml_compute_plan.py --model=model.mlpackage

# List the op types that did NOT get assigned to the ANE, grouped by type.
python coreml_compute_plan.py --model=exported_model.pte --show-non-ane

# List op types per device (ANE / GPU / CPU) — shows the delegated (ANE) ops too.
python coreml_compute_plan.py --model=exported_model.pte --show-ops

# Explore a different setting (choices: all, cpu_and_ne, cpu_and_gpu, cpu_only).
python coreml_compute_plan.py --model=exported_model.pte --compute-units=all
```

By default the plan uses the `compute_units` read from the `.pte`'s CoreML compile spec, falling back to `cpu_and_ne` when unavailable. If partitions disagree, it warns and uses the first sorted setting for all partitions. The chosen value is printed on the first line. The plan reflects the current host; confirm on-device behavior separately. Pass `--compute-units` to override the analysis setting.

Example output (`--show-ops` itemizes each device, including the ANE-delegated ops):

```
Compute units: all (model compile spec)

=== model_1 ===
  ANE:   119 / 368 ( 32.3%)
  GPU:   245 / 368 ( 66.6%)
  CPU:     4 / 368 (  1.1%)

  ANE op types (119):
       24  ios18.linear
       17  ios18.reshape
       ...
```

**NOTES**:

- Device assignment depends on the runtime/hardware and on the compute units. The plan reflects what the CoreML runtime *would* choose on the current machine; results can differ across macOS/iOS versions and devices.
- The ANE is **fp16-only**: a `float32` model will never dispatch to the ANE regardless of `--compute-units`. With `cpu_and_ne` its ops fall back to the CPU; allow the GPU (`all`) and they move to the GPU — which is typically why a CoreML model still outruns CPU-only XNNPACK even with no ANE usage.
