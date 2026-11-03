---
name: vulkan-export
description: Vulkan backend knowledge for exporting ExecuTorch models — inline VulkanPartitioner lowering, fp16 compute, the 3D-texture limit, dynamic-via-resize, mixed-delegate fusion, the source-built fork env the exports run in, and checking which shaders the shipped runtime lib has.
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
