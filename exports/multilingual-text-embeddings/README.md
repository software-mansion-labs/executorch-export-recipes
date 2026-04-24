# Multilingual text-embedding export

How to re-export the two multilingual sentence-transformer models shipped to
react-native-executorch:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `sentence-transformers/distiluse-base-multilingual-cased-v2`

> **Heads up — paraphrase-multilingual-MiniLM-L12-v2 is not usable in RNE
> today.** Its tokenizer is Unigram + Precompiled normalizer + Metaspace
> decoder, none of which `executorch/extension/llm/tokenizers` supports in the
> shipped runtime (only BPE / WordPiece / BertNormalizer / TemplateProcessing
> strings are linked in). The export still succeeds and the numbers are
> correct, but `loadTokenizerModule` fails with error code 167 at runtime.
> Unblocking this needs an upstream change in
> [pytorch/extension/llm/tokenizers](https://github.com/pytorch/executorch/tree/main/extension/llm/tokenizers).

## Prerequisites

1. Conda env with ExecuTorch 1.2.x installed from source (e.g. the
   `executorch` env used for the rest of the RNE work). A plain
   `pip install executorch==1.2.0` also works for exporting — you only need
   a custom build for **profiling** (see the profiler section).

2. Python dependencies (add on top of the base ExecuTorch env):

   ```bash
   pip install transformers huggingface_hub safetensors "optree>=0.13.0"
   ```

   `optree` is required because PyTorch nightly pulls it in for pytree
   handling even though it's listed as optional.

3. The scripts expect to reach HuggingFace to download weights and the
   `2_Dense/model.safetensors` projection head for distiluse.

## Running the export

From this directory:

```bash
# distiluse — XNNPACK fp32, the variant RNE actually consumes
python distiluse_base_multilingual_cased_v2.py \
    --backend xnnpack --precision fp32 --output-dir /path/to/out

# paraphrase — same interface, but see the "not usable in RNE" note above
python paraphrase_multilingual_minilm_l12_v2.py \
    --backend xnnpack --precision fp32 --output-dir /path/to/out
```

Each script writes two files to `--output-dir`:

| File | Purpose |
|------|---------|
| `<model>_xnnpack_fp32.pte` | ExecuTorch program |
| `<model>_tokenizer.json`   | HF fast tokenizer dump (prefixed name so multiple models can share one dir) |

At the end the script runs the program through the ExecuTorch Python runtime
and asserts `torch.allclose(runtime_out, eager_out, atol=1e-4)`. If that
passes the export is bit-equivalent to the PyTorch reference.

### Supported (backend, precision) matrix

```
xnnpack + fp32    baseline
xnnpack + 8da4w   Int8 dynamic activation + Int4 weight via torchao
coreml  + fp32    baseline
coreml  + fp16    FLOAT16 compute_precision
```

Unsupported combos are rejected by the CLI with a friendly error. Reasons:

- `xnnpack + fp16`: `model.to(torch.float16)` produces NaN at runtime
  (softmax/LayerNorm overflow).
- `coreml + 8da4w`: coremltools has no MIL mapping for the `torch.int8`
  tensors torchao emits.

### Embedding quantization (not shipped)

`torchao.IntxWeightOnlyConfig` on `nn.Embedding` emits
`torchao::dequantize_affine`, which has no out-variant in ExecuTorch 1.2 —
the exporter fails at `.to_executorch()` with
`RuntimeError: Missing out variants: {'torchao::dequantize_affine'}`. The
Llama path in `examples/models/llama/source_transformation/quantize.py`
avoids this with a hand-rolled `EmbeddingQuantHandler` but porting that is a
separate piece of work.

## Why the distiluse script looks the way it does

A few non-obvious design choices — all driven by profiling with the
executorch-profiler (section below):

- **`attention_mask=None` is passed to the transformer.** RNE tokenizes a
  single string per call and never pads, so the mask is always all-ones.
  Passing it triggers HF's `masking_utils.create_bidirectional_mask` →
  `torch.where(bool_mask, 0.0, -inf)` → a chain of `where`/`eq`/`any`/
  `logical_not`/`expand_copy` ops that XNNPACK can't delegate. We still
  concatenate CLS/SEP into `attention_mask` for future-proofing; the
  transformer just doesn't see it. This trims ~10 ms off the forward
  (48 ms → 35 ms) without changing the output.

- **Mean pooling uses the explicit `sum / clamp(sum_mask, 1e-9)` form**,
  not `Tensor.mean(dim=1)`. `mean` lowers to `native_call_mean.out`, which
  XNNPACK does **not** delegate — switching to it regresses the delegation
  ratio by ~6 pp. The explicit form decomposes to XNNPACK-friendly
  mul/sum/div.

- **Forward returns a 1-tuple.** A plain single-tensor return creates a
  `LeafSpec` output spec. A PyTorch-nightly bug means a newly constructed
  `LeafSpec` instance has no `.type` attribute (despite the dataclass field
  declaring it), and `torch.export` then crashes at
  `_decompose_and_get_gm_with_new_signature_constants`. Returning a tuple
  forces a `TreeSpec`, which has `.type` populated, and sidesteps the bug.

- **The script monkey-patches `torch.utils._pytree.LeafSpec`** at import
  time (`__new__` pre-sets `type=None`, `_context=None`, `_children=[]`;
  `__getattr__` returns those defaults for instances that skipped `__new__`
  — e.g. deepcopy reconstructs via `__reduce__`). Without this patch,
  downstream deepcopy paths inside the edge-lowering pass still hit the
  same AttributeError on `.type`. When PyTorch ships a fix, the patch
  becomes a no-op (the guard clause exits early).

## Verifying delegation % with executorch-profiler

This is optional but recommended — it tells you how much of the program is
actually running on XNNPACK vs portable CPU kernels.

### 1. Build ExecuTorch with event-tracer enabled

Event tracing is off in the default `pip install executorch`, so you need
a source build. **Do it in a clean worktree** to keep WIP branches clean,
and put it in a directory named **exactly** `executorch` — the build refuses
any other name ([pytorch/executorch#6475](https://github.com/pytorch/executorch/issues/6475)).

```bash
# Clean worktree at whatever commit matches your release branch
git worktree add --detach /abs/path/to/executorch-clean/executorch <commit>
cd /abs/path/to/executorch-clean/executorch
git submodule update --init --recursive
```

Then build. On macOS with Apple clang 21, `flatcc` trips `-Werror` on
modern diagnostics (`implicit-int-conversion-on-negation`,
`unterminated-string-initialization`). Patch
`third-party/CMakeLists.txt` to turn that off for both the
`ExternalProject_Add(flatcc_ep ...)` block **and** the
`add_subdirectory(flatcc)` block:

```diff
 set(FLATCC_DEBUG_CLANG_SANITIZE OFF CACHE BOOL "")
+set(FLATCC_ALLOW_WERROR OFF CACHE BOOL "")
 set(FLATCC_INSTALL OFF CACHE BOOL "")
 add_subdirectory(flatcc)
```

```diff
   CMAKE_ARGS -DFLATCC_RTONLY=OFF
              -DFLATCC_TEST=OFF
              -DFLATCC_REFLECTION=OFF
              -DFLATCC_DEBUG_CLANG_SANITIZE=OFF
+             -DFLATCC_ALLOW_WERROR=OFF
              -DFLATCC_INSTALL=ON
```

Then install:

```bash
CMAKE_ARGS="-DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_COREML=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
  -Dprotobuf_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release" \
./install_executorch.sh
```

`EXECUTORCH_BUILD_COREML=OFF` matters — the `pybind` preset pulls CoreML
in by default and it requires the `coremltools` nested submodule, which
isn't initialized by a plain recursive submodule update.

### 2. Run the profiler

```bash
git clone git@gl.swmansion.com:swm-ai/executorch-profiler.git
cd executorch-profiler
pip install colorama
python profile_et.py --model=<path>/<model>_xnnpack_fp32.pte --output=.
```

The profiler fabricates random integer inputs that satisfy the model's
dynamic shape, runs one forward, writes `etdump.etdp`, and prints a
per-op breakdown with a delegation percentage column.

### Reference numbers (distiluse, MacBook, random 128-token input)

```
Baseline wrapper                          90.27% delegated (43.4 / 48.1 ms)
+ attention_mask=None to transformer      89.33% delegated (31.4 / 35.2 ms)
```

Absolute wall-time drops meaningfully (~13 ms / ~27% faster forward). The
ratio stays flat because both the delegated and non-delegated slices of
time shrink — the savings come from HF no longer preparing a 4D mask.

### What's left non-delegated, and why

| Op | Source |
|----|--------|
| `native_call_where.self_out`        | `transformers/masking_utils.py:531` — `torch.where(bool_mask, 0.0, -inf)` |
| `native_call_eq.Scalar_out`, `any.out`, `logical_not.out` | rest of `create_bidirectional_mask` |
| `native_call_native_layer_norm.out` | XNNPACK's `StaticLayerNormalization` doesn't match the decomposed `native_layer_norm` pattern |
| `native_call_expand_copy.out`       | part HF mask broadcast, part the `expand` inside `mean_pooling` |

Getting past ~91% would need either surgery on HF's `masking_utils` or an
XNNPACK-side LayerNorm fusion pass. Neither is trivial and both are
separate from this export.

## Artifacts produced for RNE

For the local-testing flow documented in
`packages/react-native-executorch/src/constants/modelUrls.ts` (look for the
`LOCAL_DEV_HOST` block), serve these four files over
`python -m http.server 8000`:

```
<out>/distiluse-base-multilingual-cased-v2_xnnpack_fp32.pte
<out>/distiluse-base-multilingual-cased-v2_tokenizer.json
<out>/paraphrase-multilingual-MiniLM-L12-v2_xnnpack_fp32.pte     # still blocked by tokenizer
<out>/paraphrase-multilingual-MiniLM-L12-v2_tokenizer.json       # still blocked by tokenizer
```

Once the HF repos under `software-mansion/react-native-executorch-<slug>`
are populated, the `LOCAL_DEV_HOST` block gets swapped for the commented
`URL_PREFIX` lines in the same file.
