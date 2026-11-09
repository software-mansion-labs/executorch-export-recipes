# gemma4_omni notes

Gemma 4 E2B omnimodal (vision + audio + text) MLX export, onboarded from the
executorch repo's `export_gemma4_omni_mlx_v3.py`. Published contract:
`software-mansion/react-native-executorch-gemma-4-multimodal` (`e2b/mlx/config.json`,
4 methods: token_embedding/text_decoder/vision_encoder on MLX, audio_encoder on XNNPACK).

The published config declares the canonical `image-text-to-text` and
`audio-text-to-text` capabilities (these are in the spec enum so the spec loads).
Producing the `.pte` needs an executorch source build with the MLX backend; the
published `.pte` already exists on HF.

## Export settings

`group_size=64` (the v3 recipe's CLI default is 32, but the published multimodal artifact
was produced at gs=64 — same on-device-tested setting as the text `gemma4`). int4
weights; `max_seq_len=2048` (the published model's context). These are baked into
`export.py` to match the shipped `.pte`.

## Self-containment

The full gemma4 model + export code is vendored under `_gemma4/` (the encoders,
omni v2/v3 builders, speech_transform, quant_utils, and the text_decoder package).
Cross-imports were rewritten from `executorch.examples.models.gemma4.X` to relative
(`from .X import ...`), so there is NO `executorch.examples.models.gemma4` coupling.

The only irreducible executorch dependencies are the compiled backends, resolved at
export time in an ET build:
- `executorch.backends.mlx.*` (MLXPartitioner, quantize_model_, passes, custom_ops, llm.cache)
- `executorch.backends.xnnpack.*` (audio_encoder partitioner)
- `executorch.extension.llm.custom_ops`, `executorch.exir.*`

## Known residual (acceptable)

`_gemma4/quant_utils.py:apply_embedding_quantization` imports
`executorch.examples.models.llama.source_transformation.quantize.EmbeddingQuantHandler`.
This is a LAZY import inside that function, and the function is only called on
`export_gemma4.py`'s XNNPACK embedding-quant path — NOT the omni MLX export path
(v3 uses `executorch.backends.mlx.llm.quantization.quantize_model_`). So it is dead
code for this model, does not affect import-safety, and was left verbatim rather than
editing vendored model logic. If a fully example-decoupled vendor is ever wanted,
inline EmbeddingQuantHandler or drop the unused embedding-quant path.

## Core ML: not supported, and not attempted

This model has **no Core ML export and is not expected to get one.** Apple-silicon
acceleration here is MLX only (see `mlx/export.py`), which additionally requires the
software-mansion-labs ExecuTorch fork -- the PyPI wheel produces an MLX payload the
RNE runtime cannot load.

**Why Core ML is not on the table.** This is an autoregressive decoder driven by a
mutable KV cache. ExecuTorch's Core ML backend has no practical path for that: the
cache is a mutable buffer carried across calls, and the decode step needs a growing
sequence dim. Core ML's dynamic-shape support is the exact area where we already
have a confirmed, filed defect -- non-CPU compute paths reject shapes Core ML infers
as data-dependent (apple/coremltools#2825, see `docs/coremltools-2825-repro.md`).

**Status: a judgment call, not a measured result.** Nobody has run this model through
the Core ML backend and collected a failure. It is not attempted because the cost is
high and the prior is bad, not because it was tried and rejected. If Core ML gains
usable stateful-KV-cache support, this is worth revisiting.

## Vulkan: packed CPU embeddings for issue #1613

The Vulkan omni recipe now keeps the entire `token_embedding` method portable.
Its partitioner list is empty, allowing `do_quant_fusion_and_const_prop=True`
to produce `quantized_decomposed::embedding_4bit.dtype_out` CPU kernels with
packed weights. Passing this method to Vulkan uploads the large embedding
tables during load; passing it to XNNPACK can fold dequantized tables into a
delegate. The decoder and vision methods still use Vulkan, and audio still
uses XNNPACK. The text-only `forward` recipe is unaffected.

This restores CPU lookup without reverting the new runner contract: outputs
remain float32, sequence bounds and schema stay the same, and main/PLE
quantization remains 4-bit. The older 2-bit PLE setting is not restored:
`gemma4/_gemma4_edge/export/quant.py` records multimodal quality problems with
that setting. The change is marked `EXPORT PATCH` in the vendored pipeline.

The motivating measurements used the published artifacts in fresh native
processes on a Pixel 10 with the same RNE `nativeLibsVersion=0.11.0` libraries:

| Published embedding method | GL mtrack (MiB) | Total PSS (MiB) |
| --- | ---: | ---: |
| v0.9.0: CPU quantized lookups and small Vulkan subgraph | 439.62 | 589.59 |
| v0.11.0: Vulkan lookups | 1650.89 | 1720.34 |

Both executed token `[[0]]`. These are isolated-method, post-execution samples,
not full-model load peaks or measurements of the new candidate. Artifact hashes:

- v0.9.0 `react-native-executorch-gemma-4-multimodal`, SHA256
  `56c6137e47ae5b64174259deb5d96a5d18bb86f2d992cfd96b65d869889b3fd2`.
- v0.11.0 `gemma-4-E2B-it-ExecuTorch`, SHA256
  `e7ce806617d2da228e398302ccd69b566898f621416f37a5b622c91974301652`.

### Candidate verification (2026-11-09)

Run the checkpoint-free check in a Gemma Vulkan export environment:

```bash
python tools/verify_gemma_cpu_embeddings.py
```

The check uses the actual Gemma model, quantization, fp16 retarget, omni float32
wrapper, and lowering pipeline with a small deterministic configuration
(vocabulary 128, hidden size 64, two layers, PLE dimension 32). It inspects the
serialized PTE for packed 4-bit CPU kernels, no delegates and no remaining
`dequantize_affine` operations. It executes sequence lengths 1, 8, 128, 3, 1
and requires exact output equality with the quantized eager reference, including
float32 dtype, shape and finite values.

Passed on macOS arm64 with Python 3.10.19, torch 2.14.0, ExecuTorch 1.4.1,
torchao 0.18.0 and transformers 5.14.1. The small PTE is 18,464 bytes and the
maximum absolute output error is zero. This diagnostic environment uses stable
torch 2.14.0; the full recipe's fork/nightly environment remains to be checked.

The initial small-model check was followed by the full export and native Pixel
checks below. The strict decoder/eager parity gate remains unmet, so the recipe
is still marked `validated_on_device=False`. Subsequent S24 verification and
user-approved v1.0.0 publication are recorded below.

Re-export in the matching fork environment with a local source checkpoint:

```bash
GEMMA4_CHECKPOINT=/path/to/checkpoint PYTHONPATH=src python -m exporter.cli \
  export gemma4_omni --backend vulkan --size e2b \
  --output-dir exports/gemma4-omni-cpu-embeddings
```

Validate the resulting artifact following
[the export validation skill](../../skills/validate-and-optimize/SKILL.md)
before replacing the published model. See
[RNE issue #1613](https://github.com/software-mansion/react-native-executorch/issues/1613).

### Full export and Pixel 10 validation (2026-11-09)

Generated `exports/gemma4-omni-cpu-embeddings-1613-fork/gemma_4_e2b_vulkan_8da4w.pte`
in 1007.6 seconds: 4,007,807,616 bytes (3.73 GiB), SHA256
`c08eadeea409d198a59b319fc8f09ef3ea5c2f965248322f527147ca67d1cf9d`.
The directory is gitignored and contains the reference tensors, validation
JSON, source/recipe provenance and native probe sources. Raw logs are under
`logs/gemma4-*-1613*.log`.

The source checkpoint is `google/gemma-4-E2B-it` revision
`4adf15748de9e1a0097b6274cee9c15836c41c37`. Ahead-of-time Python code came from
`software-mansion-labs/executorch` commit
`0a0871900c8f63a032981ac4748b9cdda84bfcb8` on `@ms/separate-backends-1.4.1`,
packaged with `setup.py build_py` and supplemented with native Python extensions
and resources from the 1.4.1 wheel. The Python/package versions are the same as
the small check above. This is not a fresh native build of that fork; Android
tests used the RNE `nativeLibsVersion=0.11.0` binaries. Exact versions, commands
and the dirty recipe diff are saved in the artifact's `provenance/` directory.

The full export was invoked through `export_variant` for
`gemma4_omni / vulkan / e2b`, with hooks saving fixed unquantized and quantized
eager embedding and prefill/decode fixtures. The reproducible controller is
`provenance/run_full_export.py`; its environment is recorded in
`provenance/environment.json`.

Checks that passed:

- The serialized embedding method has no delegates or unfused
  `dequantize_affine` operators and uses packed CPU 4-bit kernels. Decoder and
  vision use Vulkan; audio uses XNNPACK.
- All four methods' serialized upper bounds and dtypes agree with
  `get_model_schema`. Context 2048, sequence bound 128, vocabulary 262144,
  KV-cache/dynamic-shape flags and EOS IDs also match the declared contract.
- Full-artifact CPU embedding execution is exactly equal to quantized eager
  output on both the Mac and Pixel at lengths 1, 8 and 128, including pad/media
  IDs and repeat execution. Outputs have the expected float32 shapes and are
  finite.
- All model methods loaded on the Pixel. Prefill, teacher-forced decode, KV
  reset and a chat generation executed successfully. Candidate and published
  Vulkan models generated the same token IDs for the Germany-capital prompt:
  `Berlin.<turn|>`. This is a smoke check, not a task-quality evaluation.
- Feeding the published decoder the exact candidate embedding tensors gives
  bit-for-bit identical prefill, decode and reset logits to the candidate.

Single-run native memory snapshots (`dumpsys meminfo`, GiB):

| Measurement | Published v0.11 Vulkan | CPU embedding candidate |
| --- | ---: | ---: |
| Total PSS after all methods loaded | 4.71 | 3.36 |
| GL mtrack after all methods loaded | 4.30 | 2.69 |
| Total PSS after the inference fixtures | 4.73 | 3.85 |

The Pixel has 12 GB RAM. Phone storage was insufficient for the complete
candidate as one file: the initial ordinary transfer failed and its partial
file was removed. The candidate probe therefore used the **complete, unchanged
model bytes**, split at a segment boundary into a 3,736,632,448-byte mmap file
and a 271,175,168-byte tail received into aligned RAM. Absolute candidate PSS
above includes that extra 258.6 MiB; it is not subtracted from the measurements.
The published model fit as one ordinary mmap file. These are post-load samples,
not definitive transient peaks, an app benchmark, or proof of the S24/8 GB case.
No existing phone files were moved. Temporary device artifacts were cleaned up.

**Numerical gate still unmet:** the predeclared decoder/eager cosine minimum
was 0.99. Both models fall below it on the fixed eight-token fixture:

| Logits vs quantized eager | Published v0.11 Vulkan | CPU embedding candidate |
| --- | ---: | ---: |
| Prefill cosine | 0.976261 | 0.984714 |
| Decode cosine | 0.922859 | 0.979479 |

Both retain the reference argmax at those steps. The published GPU embedding's
PLE output is exactly equal to the CPU reference; main embedding relative L2
error is about 0.00021. With controlled identical embeddings the decoder logits
are bit-identical, isolating the end-to-end difference on these fixtures to
embedding rounding rather than a change in decoder computation. The cosine
budget was not relaxed after observing the results.

Independent unquantized Transformers references were also captured. The
unquantized vendored port agrees closely with Transformers (cosines 0.999885
prefill / 0.998688 decode); quantization itself contributes additional drift
(quantized eager vs Transformers: 0.939047 / 0.880723 on this fixture). These
diagnostics do not establish a broad generation-quality budget.

The memory change and targeted embedding/contract checks passed. Keep this
artifact as a candidate until the decoder/eager gate or an explicitly justified
task-quality budget is resolved. Remaining coverage includes image/audio forward parity, representative task
quality and steady-state latency. The S24 app checks below cover ordinary
single-file loading on an 8 GB device. See
`validation-summary.json`, `candidate-device-validation.json`,
`published-device-comparison.json`, `controlled-decoder-comparison.json` and
`huggingface-reference-comparison.json` in the local artifact directory.


### S24 and NLP demo confirmation (2026-11-09)

Tested the complete candidate as an ordinary single file on the connected
8 GB Samsung S24 (SM-S921B, Android 17). Three fresh standalone native runs
passed all-method loading, exact CPU embedding fixtures and text generation.
Loaded PSS was 3.196–3.215 GiB versus 4.797 GiB for published v0.11;
graphics memory was 2.854 versus 4.458 GiB. Both passed the standalone probe,
so that probe alone did not reproduce the foreground-app kill.

The actual `apps/nlp` LLM demo used its existing `useLLMChatSession` path,
Gemma Vulkan configuration, tools and default generation settings, with only
the model/tokenizer paths temporarily redirected to local files. The cached
debug APK was installed and served current sources through Metro. Its native
library hashes differ from the standalone probe; both app models used the
same APK, with provenance recorded in `s24/nlp-demo/apk-provenance.json`.

The candidate loaded, answered “Berlin.”, then handled a second turn by calling
`calculate(a=2,b=2,operation=add)` and replying “The result of two plus two is
4.” The first adb-entered question accidentally ended in `briefly.o`; the
exact prompt is retained in the logs. Sampled candidate app PSS peaked at
3.988 GiB; this is a sampled value, not a guaranteed transient peak.

The published v0.11 model was then loaded through the same demo in a fresh
process. Android killed it before the model became ready. `lmkd` identified
PID 4544 at `oom_score_adj 0` and reported “min2x watermark is breached even
after kill”; `dumpsys activity exit-info` recorded `LOW_MEMORY`. A fresh
candidate app load after this failure also reached ready successfully.
This reproduces the reported load failure with the published model and shows
the embedding change resolves it in this S24/demo test. It does not resolve
the separate numerical-quality gate documented above. Image/audio methods
were loaded, but their forward outputs were not tested.

Raw native results, demo logs, memory samples, screenshot and temporary test
configuration are under the local artifact directory's `s24/` folder.


### v1.0.0 publication (2026-11-09)

After reviewing the fixed-fixture numerical results and successful S24 demo
checks, the user explicitly approved publication. Published the exact tested
artifact (SHA256 `c08eadeea409d198a59b319fc8f09ef3ea5c2f965248322f527147ca67d1cf9d`)
to `software-mansion/gemma-4-E2B-it-ExecuTorch` at `v1.0.0`, commit
`5e5ac24fdc695a93538b046c52c036acafb47110`. The tag is based on the previous
main commit `c8d43aefc6f79054d181c09bcf9f27214787e450` with only
`e2b/vulkan/gemma_4_e2b_vulkan_8da4w.pte` replaced. Remote manifests confirm
all other files, previous tags and main are unchanged. Local publication
manifests and checks are saved under the artifact's `publication/` directory.
The numerical threshold and unmet checks above remain recorded unchanged;
publication approval does not establish broader task-quality verification.
