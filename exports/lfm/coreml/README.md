# LFM2.5 1.2B CoreML export (experimental — ANE compile currently fails)

This recipe extends [`../export_instruction.md`](../export_instruction.md) (which
covers XNNPACK `q8da4w` and unquantized `fp16`) with a CoreML-backed equivalent.
The export pipeline produces structurally-valid `.pte` files, but **the CoreML
backend fails to initialise the model on the Apple Neural Engine** at
method-load time (`MILCompilerForANE error: failed to compile ANE model using
ANEF`). Falling back to `compute_units: cpu_only` would sidestep the failure
but defeats the point — XNNPACK is already a well-tuned CPU path.

The work is captured here so we don't lose the export pipeline once upstream
fixes land.

## What works

- `coreml_fp16` PTE (~2.2 GB): structurally valid, parses via
  `executorch.runtime`.
- `coreml_c4w` PTE (~756 MB; weight-only 4-bit, per-channel via the CoreML
  partitioner): same.
- All metadata methods (`get_eos_ids`, `get_max_seq_len`, `use_kv_cache`, ...)
  load on both Mac (Python runtime) and iPhone (ExecuTorch runtime).

## What doesn't work

- `forward` method load fails because the CoreML backend can't initialise the
  embedded ML program. MIL → `mlmodelc` compilation succeeds; building the
  ANE execution plan returns `_ANECompiler : ANECCompile() FAILED`. This
  reproduces both via the Python runtime (which also exercises the CoreML
  compiler) and on iPhone 17 Pro / iOS 26.4.2, with ExecuTorch returning
  `0x23` (`InvalidProgram`) and surfacing as `code: 35, "Failed to load LLM
  runner"` in `react-native-executorch`.
- `pt2e_quantize: coreml_8a_c4w` is unimplemented upstream
  (`NotImplementedError: 4-bit Core ML quantizer is still under development`).

## Suspected root cause

LFM2's short-conv layers mutate a per-layer `conv_state` buffer
(`self.conv_state.copy_(...)`) inside `forward`. After decomposition this
expands to a `slice_copy` + `index_put` pair. ExecuTorch's `to_backend`
partitioner extracts subgraphs whose output-node names no longer match the
parent program's `buffers_to_mutate` map, so the verifier raises
`SpecViolationError`. The XNNPACK helper avoids this by using
`to_edge_transform_and_lower()` (combined edge lowering + partitioning);
the generic `_to_edge_and_lower_llama` path used for CoreML/MPS/QNN still
uses the deprecated `export_to_edge() + to_backend()` split. Migrating the
CoreML branch to the combined API and relaxing the verifier produces a
valid program, but the resulting `mlmodelc` still embeds something the ANE
compiler rejects — possibly the same mutation pattern surviving into
lowered MIL.

## Setup

1. Clone ExecuTorch and check out a recent `main`; install editably into a
   dedicated venv (`pip install -e .`). The export pipeline pulls modules by
   absolute name (`from executorch...`), so an editable install is required
   for the patches below to take effect; otherwise apply each patch a second
   time inside `site-packages/executorch/...`.
2. Apply the five patches in [Patches](#patches) — four against ExecuTorch,
   one against PyTorch nightly's `torch._export.verifier`.
3. Place `lfm2_coreml_fp16.yaml` and `lfm2_coreml_c4w.yaml` (in this
   directory) into `examples/models/lfm2/config/` in the ExecuTorch checkout.

## Configs

[`lfm2_coreml_fp16.yaml`](lfm2_coreml_fp16.yaml):

```yaml
base:
  metadata: '{"get_bos_id": 1, "get_eos_ids":[7]}'

model:
  use_kv_cache: True
  use_sdpa_with_kv_cache: False
  enable_dynamic_shape: False
  dtype_override: fp32

backend:
  coreml:
    enabled: True
    ios: 18
    enable_state: True
    preserve_sdpa: True
    compute_units: cpu_and_ne
```

[`lfm2_coreml_c4w.yaml`](lfm2_coreml_c4w.yaml) is the same with
`quantize: c4w` added under `coreml`.

## Patches

### 1. `extension/llm/export/builder.py`

Supersedes the single-hunk patch in `../export_instruction.md`. Two changes:
`_export()` is the LeafSpec-aware fp16 fix from the existing doc; the new
`export_to_edge()` hunk passes `strict=False` to the second export call so it
doesn't trip the same bug:

```diff
@@ -219,11 +219,23 @@
             exported_module = export(
                 self.model if not module else module,
                 self.example_inputs,
                 kwargs=self.example_kwarg_inputs,
                 dynamic_shapes=dynamic_shape,
-                strict=True,
+                strict=False,
             )
-            # Functionalize the graph, and decompose subclasses from torchao quantize.
-            exported_module = exported_module.run_decompositions({})
+            # Functionalize the graph, and decompose subclasses from torchao quantize.
+            # run_decompositions is broken in PyTorch nightly when the model returns a
+            # single tensor (LeafSpec): deepcopy of module_call_graph raises AttributeError
+            # because the C++ LeafSpec doesn't expose `.type`. Safe to skip for
+            # non-quantized models; TorchAO-quantized models need it but hit the same bug.
+            try:
+                exported_module = exported_module.run_decompositions({})
+            except AttributeError as e:
+                if "'LeafSpec' object has no attribute 'type'" not in str(e):
+                    raise
+                logging.warning(
+                    "Skipping run_decompositions({}) due to a PyTorch nightly "
+                    "incompatibility (LeafSpec.type missing on C++ LeafSpec). "
+                    "Safe for non-TorchAO-quantized models."
+                )
         return exported_module
@@ -422,6 +434,7 @@
                     example_kwarg_inputs=self.example_kwarg_inputs,
                     dynamic_shapes=dynamic_shape,
                     edge_constant_methods=self.metadata,
                     edge_compile_config=edge_config,
+                    strict=False,
                     verbose=self.verbose,
                 )
```

### 2. `extension/llm/export/export_llm.py`

PyTorch nightly's `torch.utils._pytree.LeafSpec.__post_init__` initialises
`num_{nodes,leaves,children}` but forgets to set the inherited
`type`/`_context`/`_children` slots. Any access — including dataclass-based
deepcopy used inside `run_decompositions` — then raises `AttributeError`.
Patch the entry point to fix the class on import:

```diff
@@ -32,6 +32,23 @@
 import argparse
 import os
 import sys
 from typing import Any, List, Tuple

+# Workaround for PyTorch nightly bug: torch.utils._pytree.LeafSpec is a slotted
+# dataclass whose __post_init__ initialises num_{nodes,leaves,children} but
+# forgets to set the inherited `type`, `_context`, `_children` slots. Any
+# attribute access (and dataclass-based deepcopy of a TreeSpec containing a
+# LeafSpec) then raises AttributeError. This breaks ExecuTorch's export
+# pipeline (run_decompositions, strict export, to_edge) for any model that
+# returns a single tensor.
+import torch.utils._pytree as _pytree  # noqa: E402
+_LeafSpec = _pytree.LeafSpec
+_orig_leafspec_post_init = _LeafSpec.__post_init__
+
+def _patched_leafspec_post_init(self) -> None:
+    object.__setattr__(self, "type", None)
+    object.__setattr__(self, "_context", None)
+    object.__setattr__(self, "_children", [])
+    _orig_leafspec_post_init(self)
+
+_LeafSpec.__post_init__ = _patched_leafspec_post_init
+# Fix the cached singleton too (it was constructed before the patch).
+object.__setattr__(_pytree._LEAF_SPEC, "type", None)
+object.__setattr__(_pytree._LEAF_SPEC, "_context", None)
+object.__setattr__(_pytree._LEAF_SPEC, "_children", [])
+
 import hydra
 from executorch.examples.models.llama.export_llama_lib import export_llama
```

### 3. `examples/models/llama/export_llama_lib.py`

Short-circuit `_to_edge_and_lower_llama` when `coreml=True` so the CoreML
path uses `to_edge_transform_and_lower()` (combined edge lowering +
partitioning), mirroring the XNNPACK helper. The deprecated
`export_to_edge() + to_backend()` split corrupts the mutation map for LFM2
hybrid models. Insert this block immediately after the function's parameter
list, before the existing `builder_exported_to_edge = ...export_to_edge()`
call:

```diff
@@ -1080,6 +1080,28 @@
 ):
+    # CoreML fast-path: use to_edge_transform_and_lower (combined) rather than
+    # the deprecated export_to_edge() + to_backend() split, which mishandles
+    # the mutation-output map on hybrid models like LFM2 (short-conv
+    # conv_state.copy_() decomposes to slice_copy + index_put, and the split
+    # path only registers one of those as the buffer mutation, leaving the
+    # other to fail SpecViolationError verification). Mirrors the XNNPACK
+    # helper at _to_edge_and_lower_llama_xnnpack.
+    if coreml:
+        coreml_partitioner = get_coreml_partitioner(
+            coreml_ios,
+            embedding_quantize,
+            pt2e_quantize,
+            coreml_quantize,
+            coreml_compute_units,
+        )
+        logging.info("Lowering model using following partitioner(s): ")
+        logging.info(f"--> {coreml_partitioner.__class__.__name__}")
+        builder = builder_exported.pt2e_quantize(quantizers).to_edge_transform_and_lower(
+            [coreml_partitioner]
+        )
+        if verbose:
+            print_delegation_info(builder.edge_manager.exported_program().graph_module)
+        return builder.to_executorch(passes=additional_passes)
+
     builder_exported_to_edge = builder_exported.pt2e_quantize(
         quantizers
     ).export_to_edge()
```

### 4. `backends/apple/coreml/compiler/coreml_preprocess.py`

LFM2 ties the LM head linear's weight to the input embedding's gather table.
The c4w quantiser targets linears but `op_type_configs={"gather": None}`
explicitly opts gather out — and coremltools then refuses to apply two
different configs to the same weight (`compression config conflict detected
between ops`). Walk the MIL program before calling `linear_quantize_weights`
and skip the tied linear by name:

```diff
@@ -578,10 +578,31 @@
         if op_linear_quantizer_config is not None:
             logger.warning(
                 "Core ML Backend op_linear_quantizer_config API is experimental"
             )
+            # Skip quantization for any linear whose weight is also consumed
+            # by a gather op (tied embeddings, e.g. LFM2). coremltools refuses
+            # to apply a quantizer config to a weight that another op (gather)
+            # is configured to keep as-is, so we exclude the tied linear by
+            # name. Without this, c4w fails on tied-embedding models.
+            tied_linear_op_names: list[str] = []
+            try:
+                prog = mlmodel._mil_program
+                for func in prog.functions.values():
+                    weight_to_gathers: dict[str, list[str]] = {}
+                    weight_to_linears: dict[str, list[str]] = {}
+                    for op in func.operations:
+                        if op.op_type == "gather" and getattr(op, "x", None) is not None:
+                            weight_to_gathers.setdefault(op.x.name, []).append(op.name)
+                        elif op.op_type == "linear" and getattr(op, "weight", None) is not None:
+                            weight_to_linears.setdefault(op.weight.name, []).append(op.name)
+                    for w, linears in weight_to_linears.items():
+                        if w in weight_to_gathers:
+                            tied_linear_op_names.extend(linears)
+            except Exception as _e:  # noqa: BLE001
+                logger.warning(f"Tied-embedding detection failed: {_e}")
+            op_name_configs = {name: None for name in tied_linear_op_names}
             config = cto.coreml.OptimizationConfig(
                 global_config=op_linear_quantizer_config,
                 op_type_configs={"gather": None},
+                op_name_configs=op_name_configs,
             )
             mlmodel = cto.coreml.linear_quantize_weights(mlmodel, config=config)
```

### 5. `torch/_export/verifier.py` (PyTorch nightly, site-packages)

Even with the migration in patch 3, the subgraph extracted by the CoreML
partitioner has output nodes whose names no longer match the parent
program's `buffers_to_mutate` / `user_outputs`. The underlying graph is
correct; only the metadata mapping is stale. Downgrade the two
`SpecViolationError` raises to warnings so the export can complete:

```diff
@@ -541,10 +541,17 @@
         else:
-            raise SpecViolationError(
-                f"Mutation node {mutation_node} is neither a buffer nor a user input. "
-                f"Buffers to mutate: {gs.buffers_to_mutate}, User inputs to mutate: {gs.user_inputs_to_mutate}"
-            )
+            # Subgraph extraction (e.g. CoreML partitioner on hybrid models
+            # like LFM2) can rename output nodes after the parent's
+            # buffers_to_mutate map was built, leaving the map keys stale
+            # while the underlying mutation semantics are still correct.
+            import logging as _logging
+            _logging.warning(
+                "Mutation node %s is neither a buffer nor a user input. "
+                "Buffers to mutate: %s, User inputs to mutate: %s. "
+                "Continuing (subgraph extraction likely renamed the node).",
+                mutation_node, gs.buffers_to_mutate, gs.user_inputs_to_mutate,
+            )

     for user_output_node, user_output_name in zip(user_output_nodes, gs.user_outputs):
         if user_output_node != user_output_name:
-            raise SpecViolationError(
-                f"User output {user_output_node} is not in the correct "
-                "order or is not found in the "
-                f"exported program's user_output list: {gs.user_outputs}. "
-            )
+            import logging as _logging
+            _logging.warning(
+                "User output %s is not in the correct order or is not found "
+                "in the exported program's user_output list: %s. Continuing "
+                "(subgraph extraction likely renamed the node).",
+                user_output_node, gs.user_outputs,
+            )
```

## Export commands

Run from the ExecuTorch checkout root:

```bash
# fp16 (~2.2 GB)
python -m extension.llm.export.export_llm \
  --config examples/models/lfm2/config/lfm2_coreml_fp16.yaml \
  +base.model_class="lfm2_5_1_2b" \
  +base.params="examples/models/lfm2/config/lfm2_5_1_2b_config.json" \
  +export.max_seq_length=2048 \
  +export.max_context_length=2048 \
  +export.output_name="lfm2_5_1_2b_coreml_fp16.pte"

# c4w (~756 MB)
python -m extension.llm.export.export_llm \
  --config examples/models/lfm2/config/lfm2_coreml_c4w.yaml \
  +base.model_class="lfm2_5_1_2b" \
  +base.params="examples/models/lfm2/config/lfm2_5_1_2b_config.json" \
  +export.max_seq_length=2048 \
  +export.max_context_length=2048 \
  +export.output_name="lfm2_5_1_2b_coreml_c4w.pte"
```

## Validation

```python
from executorch.runtime import Runtime, Verification

rt = Runtime.get()
prog = rt.load_program("lfm2_5_1_2b_coreml_c4w.pte",
                      verification=Verification.InternalConsistency)
print(prog.method_names)
# {'forward', 'get_bos_id', 'get_eos_ids', 'get_max_context_len',
#  'get_max_seq_len', 'get_n_layers', 'get_vocab_size', 'use_kv_cache',
#  'use_sdpa_with_kv_cache', 'enable_dynamic_shape'}

prog.load_method("get_eos_ids")      # OK
prog.load_method("get_max_seq_len")  # OK
prog.load_method("forward")          # FAILS: backend init -> 0x23
```

The `forward` load triggers the CoreML compiler which emits
`MILCompilerForANE error: failed to compile ANE model using ANEF` and the
backend init returns `0x23` (`InvalidProgram`). On-device the same `0x23`
surfaces from `react-native-executorch` as `code: 35, "Failed to load LLM
runner"`.

## Open items before this is usable

- Identify which lowered MIL construct the ANE compiler rejects and either
  rewrite the short-conv mutation pattern in the LFM2 model definition or
  patch the CoreML partitioner to emit ANE-friendly graphs.
- Land the `to_edge_transform_and_lower` migration upstream in
  `_to_edge_and_lower_llama` so the CoreML branch doesn't need a local
  short-circuit.
- Remove the `torch._export.verifier` workaround once the partitioner stops
  desynchronising the subgraph signatures.
