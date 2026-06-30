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

## Environment

`coremltools` + the executorch CoreML backend must be installed (conda env had
coremltools 9). Heads-up: torch 2.12 warns it is untested with coremltools
(2.7 is the most-tested). `.pte` production has not been device-verified for
every model — the int8 quant dict and `-1`->Dim choices in particular want a
real export + on-device check.
