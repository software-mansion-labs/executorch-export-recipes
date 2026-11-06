---
name: qnn-export
description: Qualcomm QNN backend knowledge for exporting ExecuTorch models — the x86_64-Linux-only host setup (QNN SDK + ExecuTorch source build), inline HTP lowering, 8a8w PTQ, the per-SoC compile, and on-device validation.
---

# QNN Export

QNN (Qualcomm AI Engine Direct) targets the **HTP** — the Hexagon Tensor
Processor on Qualcomm SoCs. Models lower **inline** in their `qnn/export.py`.

Status: experimental. `mobilenet_v2` is the proof-of-concept
(react-native-executorch#1333); nothing is published or wired into the app yet.

## The host requirement: x86_64 Linux, not macOS

The QNN SDK ships host libraries for `x86_64-linux-clang` and
`x86_64-windows-msvc` only — there is **no macOS host library**, so the AOT
compile cannot run on an Apple Silicon Mac. Only the device-side runtime is
cross-platform. Export on Ubuntu 22.04 x86_64 (WSL works); the phone is the only
part shared with a Mac workflow.

## One-time setup on the Linux host

Prerequisites: Ubuntu 22.04 x86_64 (or WSL), `g++` >= 13, python 3.10, the
Android NDK (for the on-device runner), and **QNN SDK 2.37.0** from the Qualcomm
developer portal, unpacked to e.g. `~/qnn-sdk/qairt/2.37.0.250724` (2.37 is the
version ExecuTorch currently recommends).

ExecuTorch must be built **from source with the Qualcomm backend enabled** — the
pip `executorch` wheel does not ship the QNN AOT libraries:

```bash
git clone https://github.com/software-mansion-labs/executorch.git
cd executorch && git submodule sync && git submodule update --init

export QNN_SDK_ROOT=$HOME/qnn-sdk/qairt/2.37.0.250724
export ANDROID_NDK_ROOT=$HOME/Android/Sdk/ndk/<version>
export EXECUTORCH_ROOT=$PWD

./install_executorch.sh                  # base ExecuTorch + python deps
./backends/qualcomm/scripts/build.sh     # QNN AOT libs + android runtime
```

Every shell that runs an export needs:

```bash
export QNN_SDK_ROOT=$HOME/qnn-sdk/qairt/2.37.0.250724
export EXECUTORCH_ROOT=$HOME/executorch
source $QNN_SDK_ROOT/bin/envsetup.sh     # sets LD_LIBRARY_PATH for QNN
export PYTHONPATH=$EXECUTORCH_ROOT/..:$PYTHONPATH
```

Because that env is a native source build, QNN models are **not uv-buildable**
(same situation as the MLX models): their `pyproject.toml` records the pinned
intent, no `uv.lock` is committed, and the export runs in the ET-source env with
`PYTHONPATH=src python -m exporter.cli export …`.

## The .pte is compiled for one SoC

QNN lowering bakes a context binary for a specific chipset — there is no
portable QNN `.pte`. Read the target's chip and map it to a `QcomChipset`
member; the exports take it from `QNN_SOC_MODEL`:

```bash
adb shell getprop ro.soc.model     # e.g. SM8650
```

| `ro.soc.model` | Marketing            | Hexagon |
|----------------|----------------------|---------|
| SM8450         | Snapdragon 8 Gen 1   | v69     |
| SM8550         | Snapdragon 8 Gen 2   | v73     |
| SM8650         | Snapdragon 8 Gen 3   | v75     |
| SM8750         | Snapdragon 8 Elite   | v79     |

Full list: `backends/qualcomm/serialization/qc_schema.py`. Export each target
into its own `--output-dir`, since the variant filename carries the precision
but not the SoC. No device needs to be attached at export time.

## Recipe

```python
from executorch.backends.qualcomm.export_utils import make_quantizer
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec, generate_qnn_executorch_compiler_spec,
    get_qnn_context_binary_alignment, to_edge_transform_and_lower_to_qnn,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

compiler_specs = generate_qnn_executorch_compiler_spec(
    soc_model=getattr(QcomChipset, soc),
    backend_options=generate_htp_compiler_spec(use_fp16=not quantized),
)
if quantized:                                     # 8a8w PTQ
    captured = torch.export.export(model, sample, strict=False).module()
    annotated = prepare_pt2e(captured, make_quantizer(
        quant_dtype=QuantDtype.use_8a8w, soc_model=soc))
    for x in calibration_inputs: annotated(*x)    # activation ranges
    model = convert_pt2e(annotated)
edge = to_edge_transform_and_lower_to_qnn(model, sample, compiler_specs)
program = edge.to_executorch(config=ExecutorchBackendConfig(
    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True,
                                            alloc_graph_output=True),
    segment_alignment=get_qnn_context_binary_alignment(),
))
```

`backends/qualcomm/export_utils.py` also has a one-call `build_executorch_binary`
helper, but it names the output file itself and calls `sys.exit(0)` under
`compile_only` — which would kill the harness CLI mid-export. Lower inline with
the primitives above and write the buffer to the spec's variant filename.

## Quantization

- **8a8w** (int8 activations + int8 weights, `QuantDtype.use_8a8w`) is the
  default HTP recipe; `make_quantizer` is per-channel for conv by default.
  16a4w / 16a8w / 8a4w exist for accuracy-vs-size trades.
- PTQ needs **real calibration data**. Calibrating on random noise produces a
  `.pte` that exports and runs but predicts garbage — useful only as a toolchain
  smoke test, and worth a loud warning in the export.
- **fp16 on HTP is a compile-time precision** (`use_fp16=True`), not a
  quantization pass, and it needs a premium-tier SoC (8 Gen 1 or newer).

## On-device validation

The `.pte` is not verified until it runs on the Hexagon. The fastest end-to-end
check is upstream ExecuTorch's own reference runner, which exports, pushes, runs
on the HTP and reports top-k in one shot (needs the android
`qnn_executor_runner` from the setup step):

```bash
cd $EXECUTORCH_ROOT
python -m examples.qualcomm.scripts.mobilenet_v2 \
  -b build-android -m SM8650 -s <adb_serial> --dataset ./imagenet-mini/val
# expect roughly top_1 ~70%, top_5 ~90%
```

A healthy top-k proves the QNN compile + Hexagon execution path works on that
SoC. Then check the harness-produced `.pte` against the same reference.

## react-native-executorch integration

Loading a QNN `.pte` in the app is a **separate, larger task**: RNE's native
ExecuTorch build is XNNPACK/CoreML only and would need the QNN delegate plus the
QNN runtime `.so`s and Hexagon skels shipped in the APK. Tracked in
react-native-executorch#1333.
