# Vulkan precision and Android validation

Both variants accept RGB float32 `[1,3,320,320]` in `[0,1]` and return float32
`[1,1,320,320]` foreground logits. Input normalization stays in FP32. Each
export contains one Vulkan delegate for the network and one portable input
`amax` reduction. Texture limits are capped at 2048 in each dimension.

## FP16 with a sensitive FP32 convolution

The FP16 export fuses Conv/BatchNorm pairs before casting the network to half.
`stage2d.rebnconv1d.conv_s1` retains FP32 weights, bias and computation, with
casts around it inside the Vulkan delegate. Global `force_fp16` is deliberately
not used: it would also downcast this convolution and input normalization.

On the connected Galaxy S20+, an entirely FP16 export changed the bicycle mask
to 0.9418 IoU against XNNPACK FP32. Keeping this convolution in FP32 recovered
0.9932 IoU. The entirely FP16 candidate is not published.

## Device checks

Validation used Samsung Galaxy S20+ SM-G986B, Android 13, Mali GPU, arm64-v8a,
and the ExecuTorch 1.4.1 runtime/backend libraries shipped with RNE. A native
runner linked these exact libraries, loaded each actual `.pte`, checked float32
output type and element count, rejected non-finite output, and saved logits.
Exported method metadata was also checked against the config.

The comparison set is the same 13 images used for selective XNNPACK INT8
validation: `bike.jpg`, `boat.jpg`, `whisk.png`, plus COCO val2017 image IDs 139,
285, 632, 724, 776, 785, 802, 872, 885 and 1000. All-black and all-white inputs
were also checked. These are agreement measurements against the shipped
XNNPACK FP32 `.pte`, not ground-truth segmentation accuracy.

| Variant | Mean mask IoU (13 images) | Minimum mask IoU | File size |
| --- | --- | --- | --- |
| Vulkan FP16 with one FP32 convolution | 0.9945 | 0.9854 | 2,437,516 B |
| Vulkan FP32 | 0.999999 | 0.999986 | 4,646,540 B |

The FP16 probability MAE was at most 0.00444 across those 13 images. Both
variants produced finite outputs on all 15 inputs.

Warm native forward measurements on the same bicycle input used three warm-up
runs followed by ten measured runs per variant:

| Variant | Median |
| --- | --- |
| XNNPACK FP32 | 315.6 ms |
| XNNPACK selective INT8 | 189.1 ms |
| Vulkan FP16 with one FP32 convolution | 337.2 ms |
| Vulkan FP32 | 403.1 ms |

These were sequential host-driven device checks, not interleaved benchmarks;
they exclude model loading, image preprocessing and rendering. Vulkan does not
win on this older Mali GPU. Android retains its existing XNNPACK FP32 default;
select INT8 for the faster CPU option. Performance on other GPUs must be
measured separately. A trial with only four half-precision convolutions was
also slower (421.9 ms) and is not published.

## Reproducing the export environment

The Vulkan AOT implementation is pinned to software-mansion-labs/executorch
`@ms/separate-backends-1.4.1`, commit
`0de5ce9470e4a18d6d45585f69519267bbe218b9`. Torch is 2.14.0. The nested
`pyproject.toml` records the fork dependency; see the repository Vulkan skill
for the source-built environment used by other Vulkan models.

For AOT-only export, a compatible ExecuTorch 1.4.1 wheel can provide the common
serialization binaries while the fork supplies its Vulkan Python package.
No host GPU runtime is needed; Android validation still uses the shipped native
Vulkan library. Clone the fork into a directory named `executorch`, check out
the commit above, then from export-scripts run:

```bash
uv run --project exporter_models/u2netp python - /path/to/executorch <<'PY'
import runpy
import sys
from pathlib import Path
import executorch.backends

fork = Path(sys.argv[1]).resolve()
executorch.backends.__path__.insert(0, str(fork / "backends"))
sys.argv = [
    "exporter", "export", "u2netp", "--backend", "vulkan", "--all-variants",
    "--output-dir", "exports/u2netp/vulkan",
]
runpy.run_module("exporter.cli", run_name="__main__")
PY
```

The release exports were byte-identical to the device-validated candidates.
The Python wheel's native runtime cannot execute Vulkan on this Mac; it was
not used as a substitute for phone validation.
