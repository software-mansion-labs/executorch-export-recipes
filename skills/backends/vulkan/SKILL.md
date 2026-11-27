---
name: vulkan-export
description: Vulkan backend knowledge for exporting ExecuTorch models — lowering, precision, texture limits, dynamic shapes, mixed delegates, source builds, shader availability, and on-device delegation and per-shader profiling.
---

# Vulkan Export

Vulkan is the Android **GPU** backend for ExecuTorch. It ships inside executorch
(like XNNPACK) — no pip extra. There is **no shared `lower()` helper**; Vulkan
models lower inline in their `export.py` (same as CoreML/MLX).

Use [recover-export](../../recover-export/SKILL.md) to distinguish capture,
lowering, missing shaders, and device failures. Before keeping a precision or
graph optimization, follow
[validate-and-optimize](../../validate-and-optimize/SKILL.md): check outputs and
dynamic boundaries on the shipping runtime, then measure on the target GPU.

## Lowering

    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import to_edge_transform_and_lower
    to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner(compile_options)]).to_executorch()

Compute is **fp16** on the GPU regardless of the graph dtype (export an fp32
graph + `force_fp16`); the deliverable is effectively half-size fp16. fp16 is a
real size lever here (unlike XNNPACK, where it is compute-only).

## The 3D-texture limit (the #1 Vulkan crash)

Vulkan stores tensors as 3D textures. A tensor whose any dim exceeds the GPU's
`maxImageDimension3D` (commonly 2048) makes `vkCreateImage` fail at runtime
(Xclipse: a HostFunction crash). The classic trigger is a **large vocab
projection** (PP-OCR SVTR V=18710, multilingual recognizers). Cap it at export:

    VulkanPartitioner(compile_options={"texture_limits": (2048, 2048, 2048)})

so oversized tensors fall back instead of crashing on device. Parse the PTE
delegate flatbuffer offline to catch oversized 3D-texture tensors before shipping.

## Dynamic shapes

Vulkan does **true dynamic** via runtime resize (`require_dynamic_shapes`), fp16
only — int8 (q8ta), upsample, and scan ops do **not** resize. For OCR we ship
**static bucketed** methods instead (fully delegated, no resize footguns).

## Mixed-delegate fusion (one PTE, two backends)

Some ops crash or won't delegate on Vulkan but run fine on CPU. You can put
different methods of one fused PTE on different delegates by passing a
**per-method partitioner dict** to `to_edge_transform_and_lower`:

    parts = {m: [VulkanPartitioner(...)] for m in det_methods}
    parts.update({m: [XnnpackPartitioner()] for m in rec_methods})
    to_edge_transform_and_lower(eps, partitioner=parts)

PP-OCR ships exactly this: detect on Vulkan (fp16), recognize on XNNPACK (the
SVTR vocab head crashes Vulkan's `make_ivec2`). EasyOCR's CRNN BiLSTM also won't
delegate to Vulkan, so its recognize runs on XNNPACK too.

## Usage in export.py

Lower inline; never call the shared `exporter.backends.lower` (it is xnnpack-only
and raises for other backends). Heavy imports stay inside `export()`.

## Environment: the fork pin is not uv-buildable

Every `<model>/vulkan/pyproject.toml` pins executorch to the
software-mansion-labs fork (`branch = "@ms/separate-backends-1.4.1"`), because
the shipped Vulkan runtime carries fork-only fixes (the NOTE block in each file
says why). That pin is a **declaration, not a recipe**: `uv sync` of it fails
before compiling anything, since upstream ExecuTorch's CMake refuses a checkout
directory not named exactly `executorch` (pytorch/executorch#6475) and uv names
git checkouts by commit hash. Like the MLX models, run the export in a
source-built dev env; this section is the recipe (the MLX skill has none).

1. Clone the fork into a directory literally named `executorch`, check out the
   pinned branch, build with `EXECUTORCH_BUILD_VULKAN=ON` and install the
   package editable into that checkout's venv (the fork's
   `install_executorch.sh` / CMake flow). One checkout serves every Vulkan
   model.
2. From that venv, run the harness directly instead of `uv run --project`:

   ```bash
   cd executorch-export-recipes
   PYTHONPATH=src python -m exporter.cli \
       export <model> --backend vulkan --size <s> --output-dir <dir>
   ```

   Keep the `vulkan/pyproject.toml` (+ its `uv.lock`) as the
   record of intent — torch wheel URL, fork branch — so the stack stays
   auditable even though uv cannot materialise it.

## Profiling delegation and time per op

Rebuild the target runtime with `-DEXECUTORCH_ENABLE_EVENT_TRACER=ON`, alongside
`-DEXECUTORCH_BUILD_VULKAN=ON`, `-DEXECUTORCH_BUILD_DEVTOOLS=ON`, and
`-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON`. The tracing option defines
`ET_EVENT_TRACER_ENABLED` and enables Vulkan GPU timestamp collection; an ordinary
Vulkan build is insufficient for per-shader timings.

Follow the [Vulkan profiling workflow](../../validate-and-optimize/references/profiling.md#vulkan-on-device-delegation-and-timing)
to rebuild for Android, capture an ETDump with `executor_runner --etdump_path`,
and inspect it with `devtools/inspector/inspector_cli.py`. Inspect delegated
shader events and portable CPU fallbacks. Shader dispatch counts are not a
one-to-one count of source ops; use the lowered graph/ETRecord for source-op
coverage. Confirm timings on the intended GPU, with representative inputs.

## Running the Vulkan runtime on this Mac (MoltenVK)

The backend's C++ tests run locally on MoltenVK, no phone needed; the
generated op correctness tests (`backends/vulkan/test/op_tests`, driven by
`cases.py`) are the fastest way to prove a runtime fix. The one trap: the
runtime loads `libvulkan.dylib` through the vendored `volk`, whose macOS search
list is the bare names plus `/usr/local/lib`, so on Apple Silicon Homebrew
(`/opt/homebrew/lib`) `volkInitialize` fails and every test dies in `SetUp()`
with the misleading

    Pytorch Vulkan Runtime: The global runtime could not be retrieved because it failed to initialize.

Portability enumeration is already handled (`VK_KHR_portability_enumeration`
under `__APPLE__` in `Runtime.cpp`); you only need the loader on the dyld path:

```bash
brew install vulkan-loader molten-vk            # once
export DYLD_LIBRARY_PATH=/opt/homebrew/lib      # per shell
cmake-out/backends/vulkan/test/vulkan_compute_api_test --gtest_filter='*print_adapter'
cmake-out/backends/vulkan/test/op_tests/vulkan_op_correctness_tests --gtest_filter='*convolution_dw*'
```

Do not set `VK_ICD_FILENAMES`; the Homebrew loader finds its own ICD. Build the
op tests from their own CMake project
(`cmake -S backends/vulkan/test/op_tests -B cmake-out/backends/vulkan/test/op_tests`
then `--target vulkan_op_correctness_tests`); the generator needs
`TORCH_OPS_YAML_PATH` pointing at `torchgen/packaged/ATen/native` of the venv's
torch. MoltenVK is not an Adreno or a Mali, so it validates shader selection and
arithmetic, not device-specific behaviour (the Mali `_b1x1` branch, 3D-texture
limits).

## Shader availability: a `.pte` that lowers can still fail to load

The AOT side only records ops. The **runtime picks the GLSL shader by name when
the model loads**, from each kernel's size / stride / dilation
(`backends/vulkan/runtime/graph/ops/impl/Conv2dDW.cpp`, `Convolution.cpp`, …).
A name no `.yaml` generates raises at `loadModel`:

    Could not find ShaderInfo with name conv2d_dw_sned_half

Nothing on the export side warns about this, so **check the runtime library you
ship against, not just the export**. Shader names are plain string literals in
the generated registry, so `strings` on the Android lib lists exactly what it
ships (the RNE lib is `third-party/android/libs/executorch/<abi>/libvulkan_executorch_backend.so`
in the react-native-executorch package, built from the fork base branch
`software-mansion-labs/executorch@@ms/separate-backends-1.4.1`, the same pin
the exports use):

```bash
strings libvulkan_executorch_backend.so | grep -E '^conv2d_dw' | sort
```

Compare with the name the picker derives for the op, or just load the `.pte` on
a device. If the name is missing from **every** build (upstream's yamls included)
the picker is wrong: fix it upstream and cherry-pick to the fork branch the RNE
lib is built from — do not rewrite the model graph around it. MI-GAN's 4x4
stride-2 depthwise blur was exactly this, and digging in found a second bug
behind it: any depthwise kernel that is not 3x3 / 5x5 runs on the generic
`conv2d_dw` shader, whose tap loop lost alignment with the filter row whenever
the window crossed the right / bottom edge (any padding > 0), so even a 7x7
stride-1 depthwise conv came out wrong. Both are fixed in upstream PR
[pytorch/executorch#23892](https://github.com/pytorch/executorch/pull/23892)
(picker guard on `is_3x3 || is_5x5`, full-extent loop in `conv2d_dw.glsl`,
new depthwise cases — 4x4 s2, 7x7, dilated, non-square — in
`op_tests/cases.py`, `test_vulkan_delegate.py` and the custom-op test),
verified on MoltenVK with the recipe above; the same change is merged into
the fork's `@ms/separate-backends-1.4.1` as
[software-mansion-labs/executorch#10](https://github.com/software-mansion-labs/executorch/pull/10). migan's
`vulkanize()` graph rewrite goes away once the RNE Vulkan lib is rebuilt with
it.

## fp16 precision: the runtime, not the graph (measured on a Samsung Xclipse 940)

Echo (`exporter_models/echo/NOTES.md`, a 24-layer transformer in fp16) lost
5% argmax agreement against fp32 on a Galaxy S24 FE where MLX fp16 on an
iPhone loses 0.3%. All of it was runtime precision:

- **The Xclipse driver rounds fp16 toward zero**: fp32 -> fp16 conversions
  and fp16 arithmetic alike (2x the error of round-to-nearest-even, and
  biased, so it compounds through a residual stream). The SPIR-V execution
  mode `RoundingModeRTE` for 16-bit, which the driver advertises, fixes the
  conversions but not the arithmetic.
- **`linear_vec`, the quantized tiled linears and `custom_sdpa`'s tiled q . k
  accumulate in fp16.** fp32 accumulation costs nothing in a shared-memory
  kernel there.
- A runtime patch (fp32 accumulation, `RoundingModeRTE`, fp32 elementwise math)
  brought Echo to MLX's accuracy; it is not in the fork (Echo's NOTES.md). Its
  round-to-nearest is opt-in per program, through the `fp16_round_to_nearest`
  partitioner option: the runtime then adds the execution mode to the shader
  binaries at load, on GPUs that report support for it. Other programs run the
  shaders as built.

## Runtime gotchas found with dynamic shapes

- `llama.custom_sdpa` without an `attn_mask` is causal. Non-causal attention
  takes an additive [S, C] mask, which needs the runtime from fork 9c1e9b63c
  (react-native-executorch native libs v0.11.1); v0.11.0 aborts at load.
- The fork's `causal_sdpa` pattern asserts on a `custom_sdpa` that has no LLM
  `update_cache`; disable its detector for such graphs (Echo's `lower_vulkan`).
- `squeeze_copy`'s resize ignores negative dims (write `squeeze(2)`, not
  `squeeze(-1)`); `arange`'s resize miscounts negative steps.
- `aten.silu` and `aten.log` have no Vulkan kernel: spell SiLU as
  `x * sigmoid(x)`, take logs through `log_softmax`.
- Each method is its own delegate with its own prepacked weights and its own
  working memory sized for the bounds: a three-method program holds three
  copies of each. Echo's patched runtime (its NOTES.md) shares packed weights
  between delegates by their data hash, and the `shared_workspace` compile
  option (`VulkanPartitioner({..., "shared_workspace": True})`) gives graphs
  that never run at once one pool of working memory: int4 Echo 1.72 -> 0.61
  GB. Runtimes without the patch ignore the option. Only for graphs that keep
  no state in working memory between executions.
- The int4 linear fusion packs a weight in place at its first use, so a second
  linear on the same weight (a method that runs the network twice) stays
  unfused, as dequantize + fp16 linear. Wrapping the `quantized_linear`
  pattern's replacement so that later uses reuse the first packing works
  (Echo's NOTES.md); the proper fix is in the fork's pattern.
- flatcc's CMake writes its libraries into the source tree
  (`third-party/flatcc/lib/`), so builds for different targets overwrite
  each other's there.
