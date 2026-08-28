---
name: coreml-export
description: CoreML backend knowledge for exporting ExecuTorch models — inline CoreMLPartitioner lowering, fp16/fp32 + int8 quantization, verifying against the real shipped .pte, fully-dynamic dims.
---

# CoreML Export

CoreML is an Apple-only backend. Models lower **inline** in their
`coreml/export.py` (the shared `exporter.backends.lower` helper is xnnpack-only).
The model BUILD should be identical to the xnnpack sibling — only the lowering
differs (precision selector + partitioner). Verify by diffing the two
`export.py`: they should differ only in the lowering block.

## Recipe

```python
import coremltools as ct
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower

precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[variant["precision"]]
specs = CoreMLBackend.generate_compile_specs(
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=precision,
)
program = to_edge_transform_and_lower(
    ep, partitioner=[CoreMLPartitioner(compile_specs=specs)]
).to_executorch()
```

Multi-method (e.g. whisper encode+decode) passes a `{name: ep}` dict to
`to_edge_transform_and_lower`. Reference: the lfm2.5 retrieval embedding export
in the executorch repo uses exactly this partitioner form.

## compute_unit: pick it per method, don't take the default

`generate_compile_specs` defaults `compute_unit` to `ct.ComputeUnit.ALL`. That
default suits a big convolution/attention graph and actively hurts a small one,
because dispatching to the Neural Engine costs the same whether the program does
a lot of work or almost none.

The text embedders already act on this: `all_minilm_l6_v2` and its siblings pin
`CPU_ONLY` and record 4.1x on distiluse (33.0 -> 8.0 ms) and 3.7x on clip-text
(18.1 -> 4.9 ms). The rule generalises to any small graph, including one method
of a larger model.

Whisper is the clear case. Its `.pte` holds two methods with opposite shapes:
`encode` runs a full encoder over 30s of audio, `decode` computes one token.
Measured on an iPhone 16 (base.en, fp16), `decode` ms/step by compute unit:

| compute unit | ms/step |
| --- | --- |
| `CPU_ONLY` | 6.7 |
| `CPU_AND_GPU` | 15.3 |
| `CPU_AND_NE` | 15.8 |
| `ALL` (default) | 20.0 |

`encode` is unaffected and stays on `ALL`. End to end on 14.5s of real speech
(46 tokens), `enc-ALL / dec-CPU_ONLY` against the previously published build:
tiny 567 -> 198 ms, base 987 -> 388 ms, small 2324 -> 1558 ms, with
**byte-identical transcripts** at every size. This removes dispatch overhead; it
does not change numerics.

A multi-method model is partitioned per method, so give each its own specs:

```python
def compile_specs_for(compute_unit):
    return CoreMLBackend.generate_compile_specs(
        compute_unit=compute_unit,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )

program = to_edge_transform_and_lower(
    {"encode": ep_encode, "decode": ep_decode},
    partitioner={
        "encode": [CoreMLPartitioner(compile_specs=compile_specs_for(ct.ComputeUnit.ALL))],
        "decode": [CoreMLPartitioner(compile_specs=compile_specs_for(ct.ComputeUnit.CPU_ONLY))],
    },
).to_executorch()
```

Rule of thumb: any per-token autoregressive `decode` is dispatch-bound and wants
`CPU_ONLY`; large encoders want `ALL`. Sweep rather than assume, and sweep on
device — this is not visible from the export side.

The same sweep on a single-method model can confirm the default is right.
RF-DETR keypoint at fp16 wants `ALL` (122.7 ms) over `CPU_AND_NE` (125.0),
`CPU_ONLY` (136.1) and `CPU_AND_GPU` (171.0). Note also that the Neural Engine
is **fp16-only**: the same model at fp32 shows `CPU_AND_NE` (337.5) matching
`CPU_ONLY` (352.9) and `ALL` (173.6) matching `CPU_AND_GPU` (176.7), i.e. the
ANE contributing nothing at all. An fp32 CoreML model never reaches the ANE
whatever compute unit it asks for.

## int8 (weight quantization)

For int8 CoreML, pass `op_linear_quantizer_config` to `generate_compile_specs`:

```python
quant = {"mode": "linear_symmetric", "dtype": "int8", "granularity": "per_channel"}
specs = CoreMLBackend.generate_compile_specs(
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16,
    op_linear_quantizer_config=quant,
)
```

(Schema: `cto.coreml.OpLinearQuantizerConfig`, `weight_threshold` default 2048.)

## Do NOT carry over xnnpack-specific passes

CoreML handles addmm/linear natively. Drop xnnpack's `AddmmToLinearTransform`
and `_core_aten_ops_exception_list=[linear]` — they are partitioner-specific and
the canonical CoreML recipe uses a bare `CoreMLPartitioner`.

## The shipped HF config can be WRONG — verify against the real .pte

The published `coreml/config.json` on HF does not always match the actual
shipped artifact. Before authoring a config, read the real `.pte`:

```python
from executorch.extension.pybindings.portable_lib import _load_for_executorch
m = _load_for_executorch(pte_path)
for name in m.method_names():
    meta = m.method_meta(name)
    ins  = [tuple(meta.input_tensor_meta(i).sizes())  for i in range(meta.num_inputs())]
    outs = [tuple(meta.output_tensor_meta(i).sizes()) for i in range(meta.num_outputs())]
```

Real example: ssdlite320's shipped `coreml/config.json` declared
`[1,3,320,320] -> [1,3000,91]`, but the actual `.pte` is the same graph as
xnnpack (`[3,320,320] -> (boxes[300,4], scores[300], labels[300])`). We vendored
a corrected local config to override the bad fetched one. A config whose input
resolution doesn't match its output anchor/seq count is a red flag.

## Fully-dynamic dims

Some shipped CoreML configs use a fully-dynamic `-1` seq dim. The harness can't
store `-1`, so represent it as a concrete max with a `torch.export.Dim`:
`Dim("tokens", min=1, max=512)`. Watch for wrappers that add tokens — e.g.
paraphrase prepends CLS+SEP (+2), so the raw input dim caps at 510, and the
export example must be a small length (not the max), or torch.export raises a
constraint/`expand` error.

## Profiling & delegation check

Verifying I/O shapes (above) proves the contract; profiling proves the graph
actually lowered to the CoreML delegate rather than running on CPU portable
kernels. Use the sibling
[executorch-profiler](https://gl.swmansion.com/swm-ai/executorch-profiler) — it
runs an event-traced ExecuTorch runtime and reports per-op time plus a
**delegation ratio** per op.

```bash
# One-time: build ExecuTorch with the event tracer AND the CoreML runner.
# The --coreml flag turns on EXECUTORCH_BUILD_COREML and builds the CoreML
# executor runner; without it the delegate isn't linked in.
./build_et.sh --coreml

# Then, from the profiler dir, against your exported .pte (Apple hardware only):
python profile_et.py --model=/path/to/model.pte --output=.
```

CoreML partitions the graph into a few large delegated subgraphs, so a
correctly-lowered model shows the bulk of time under a GREEN `100%`-delegated
`DELEGATE_CALL`; ops left RED (`0%`) are the ones CoreML couldn't take and are
running on CPU — the signal to revisit the partitioner or an unsupported op.
Provide `--model-input inputs.pt` for dynamic-dim models (the random-input
fallback keys off static metadata and won't cover an `-1`->`Dim` seq axis). This
is Apple-only and, like the shipped `.pte` itself, wants a real on-device run.

## Environment

`coremltools` + the executorch CoreML backend must be installed (conda env had
coremltools 9). Heads-up: torch 2.12 warns it is untested with coremltools
(2.7 is the most-tested). `.pte` production has not been device-verified for
every model — the int8 quant dict and `-1`->Dim choices in particular want a
real export + on-device check.
