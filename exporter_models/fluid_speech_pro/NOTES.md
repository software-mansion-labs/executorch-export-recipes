# Fluid Speech Pro: full FP32 XNNPACK export

Validated on 2026-11-16 on an Apple M1 Pro, 16 GiB, macOS host CPU.
The original FP32 baseline is host-verified. The experimental INT8 follow-up
below includes a separate iOS simulator inference probe, not app integration.
The existing Gallery meeting PR has not been changed.

## Source and environment

- Model: `FluidVoice-Labs/fluid-speech-pro`, revision
  `46027a3bffc5817ce8f9b29add0a062610134a90`.
- Use the full `.nemo` reference, including `encoder.fv_insert`. The publisher's
  Transformers checkpoint omits that block and is not the export source.
- `_source_loader.py` is the unmodified publisher loader at that revision.
  Weights/source loader: CC BY 4.0, see `SOURCE_LICENSE` and `SOURCE_NOTICE`.
- Attribution: Fluid Speech Pro by FluidVoice. Built on NVIDIA Parakeet TDT
  0.6B v3 and Moondream Parakeet Ultra.
- NeMo 3.0.0 code adaptations in `_model.py` retain `NEMO_LICENSE` (Apache 2.0).
  The checkpoint's recorded training NeMo version is 2.3.0rc5; validation below
  uses the pinned, working NeMo 3.0.0 reference pipeline.
- Locked Python 3.10, PyTorch 2.14.0, ExecuTorch 1.4.1, Transformers 4.57.6.
  Exact transitive versions are in `uv.lock`.
- Raw transcription only. Clean-mode adapter weights are not enabled.

## Reproduce

From the export-recipes root:

```sh
uv sync --locked --project exporter_models/fluid_speech_pro
uv run --locked --project exporter_models/fluid_speech_pro \
  exporter export fluid_speech_pro --backend xnnpack --output-dir /absolute/output
```

The source checkpoint is downloaded anonymously at its pinned revision. To reuse
an existing download, set `FLUID_SOURCE` to a directory containing
`fluid-speech-pro.nemo`. Generated models and benchmark audio stay outside Git.
The tokenizer is the SentencePiece `.model` member of that same NeMo archive;
extract it beside the output for application decoding. Its randomized archive
prefix is not part of the model contract.

```sh
uv run --locked --project exporter_models/fluid_speech_pro python \
  exporter_models/fluid_speech_pro/validate.py \
  --pte-dir /absolute/output --source /absolute/source \
  --audio /path/IB4001.Mix-Headset.wav --start 210 --report /path/parity.json

uv run --locked --project exporter_models/fluid_speech_pro python \
  exporter_models/fluid_speech_pro/benchmark.py \
  --pte-dir /absolute/output --tokenizer /path/tokenizer.model \
  --fixture /path/fixture.json --report /path/benchmark.json

uv run --locked --project exporter_models/fluid_speech_pro python \
  exporter_models/fluid_speech_pro/inspect_artifacts.py \
  --pte-dir /absolute/output --report /path/artifact-inventory.json
```

`benchmark.py --help` describes the local fixture format. It uses fixed audio
crops, optional saved reference token IDs and historical LFM text, plus human
reference text. Source audio: AMI IB4001 headset mixture, CC BY 4.0, from
https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IB4001/audio/ .
Manual annotations: AMI public manual 1.6.2. Local campaign evidence records
reference words and exactly which historical Gallery crops were reused.
No fixture recordings or historical campaign transcripts are included in this
recipe; reproducing the table requires those saved crop/reference inputs. For
new audio, supply the following schema (times are relative to `start_seconds`):

```json
{
  "audio_path": "/path/mono-16khz.wav",
  "start_seconds": 210,
  "cohorts": [{
    "label": "example",
    "reference_text": "human transcript of all selected turns",
    "turns": [{"start": 0, "end": 5, "baseline_text": "optional prior ASR text"}]
  }]
}
```

Each crop must span 0.16–10 seconds. The recording from the offset must include
at least ten seconds for the fixed latency probes. Optional `source_tokens` on
each turn contains original NeMo greedy token IDs for exact sequence comparison.
Omit `baseline_text` unless every turn has a matched historical baseline.

## Runtime contract

Five files, all FP32. All method inputs must be contiguous.

| Component | Method | Inputs | Outputs | Bytes |
| --- | --- | --- | --- | ---: |
| preprocess | `forward` | float32 PCM `[1,160000]`, true int64 sample length `[1]` | float32 mel `[1,128,1001]`, int64 valid frame length `[1]` | 154,896 |
| subsample | `forward_0` ... `forward_7` | float32 mel `[1,128,T]`, int64 valid length `[1]` | embeddings `[1,ceil(T/8),1024]`, int64 encoded length `[1]` | 26,015,360 |
| encoder | `forward` | float32 embeddings `[1,S,1024]`, int64 valid length `[1]` | acoustic features `[1,1024,S]`, int64 valid length `[1]` | 2,459,780,352 |
| predictor | `forward` | int64 token `[1,1]`, float32 hidden/cell `[2,1,640]` each | prediction `[1,1,640]`, next hidden/cell `[2,1,640]` | 47,241,088 |
| joint | `forward` | acoustic frame `[1,1,1024]`, prediction `[1,1,640]` | raw logits `[1,1,1,8198]` | 25,289,496 |

Total PTE bytes: **2,558,481,192** (tokenizer additional).

- Subsampling supports `T=17..1001`; select `forward_{T % 8}`. Each method uses
  `T=8*groups+residue` with bounds covering its entire portion of that range.
  These eight captures share weights in one file. They preserve arbitrary
  supported crop lengths rather than padding crops to satisfy a capture guard.
- Transformer supports `S=3..126`, including true-length padding masks.
- After preprocessing, trim mel to `floor(effective_samples/160)+1` frames.
  Preserve its returned valid length, which excludes the final padded STFT frame.
- **Match NeMo's tensor transcription frontend:** for a crop shorter than one
  second, add symmetric zero padding to 16000 samples before preprocessing,
  and use that effective sample length. Then right-pad to the 160000-sample
  export capacity. The original crop's meeting timestamps stay unchanged.
- Blank/SOS token: 8192. Token logits: indices 0..8192; duration logits:
  indices 8193..8197 for durations `[0,1,2,3,4]`. Argmax each head separately.
- Initialize hidden/cell to zero independently for each crop. Advance predictor
  state only when a nonblank token is accepted. Enforce max 10 symbols per frame,
  handle zero-duration blanks, and retain no audio or state between crops.
- Raw logits deliberately omit joint-wide log-softmax; TDT has separate token
  and duration distributions. Greedy results are checked against original NeMo.
- `validate.py` contains an executable host decoding reference. No library
  modifications or new native operators were needed for these host artifacts.
- The original FP32 baseline files did not embed RNE-specific
  `get_model_schema` constants. New exports add precise range/stride domains and
  input/output constraints for encoder/subsampling via `_schema.py`. Native
  probe results for the mixed-precision candidate are documented below.

## Capture patches and recovery

The first dynamic encoder capture failed on symbolic floor/modulo guards induced
by NeMo's stride-8 subsampler and inferred views. No model weights were changed.

1. Replace the subsampler's inferred flatten dimension with its actual projection
   input width. Retain length-aware masked convolution and Fluid's original hook.
2. Capture the subsampler separately, using eight derived time dimensions to
   cover all residues. The transformer then has a direct dynamic time axis.
3. Express NeMo's relative-position shift as an equivalent padded-flat
   `index_select`, avoiding an inferred view over a noncontiguous slice. A small
   independent reproducer was bit-exact at 3, 63 and 126 frames before export.
4. Express the source LSTM's single step with its original i/f/g/o gates and
   external state, so the artifact uses supported tensor operations.

Validation compares every component to the original, unpatched eager model.
PTE serialization uses `write_to_file` rather than materializing another
multi-gigabyte `buffer` copy. Only one export/large validation ran at a time.

## Validation and benchmark

**103 component/boundary/greedy checks passed** with zero values outside the
predeclared absolute/relative budgets: preprocessing `1e-4/1e-4`; encoder,
subsampling and predictor/joint `1e-3/1e-3`. Integer lengths match exactly.
All eight subsampling methods were checked at their lower/upper bounds, with
real mel inputs. Random nonzero predictor state and 1/5/10-second greedy runs
also passed. Maximum valid encoder error on the representative full-pipeline
inputs was below `1e-6`; subsampling activations have a different numerical scale,
so their max absolute error must be read with the stated relative budget.

The complete exported pipeline matched **102/102 source greedy token sequences**
on the historical meeting crops after matching NeMo's one-second padding policy.
The initial unpadded manual pipeline differed on six short crops; diagnostic
runs showed its eager and PTE outputs agreed. The difference was the transcription
frontend's padding, not an encoder export failure. Both runs were retained locally.

Matched English text scoring lowercases and strips punctuation, preserving
number spellings and fillers:

| AMI IB4001 excerpt | Historical LFM3 edits / words | Fluid NeMo and PTE edits / words | WER, LFM3 → Fluid |
| --- | ---: | ---: | ---: |
| 210–270 s | 32 / 177 | 29 / 177 | 18.08% → 16.38% |
| 210–390 s | 108 / 497 | 94 / 497 | 21.73% → 18.91% |

These excerpts overlap and come from **one meeting**. Only crops retained in the
historical LFM transcript were reused; omitted spans are not evaluated. This is
not a representative benchmark or evidence of better diarization. No statistical
significance claim. Silence produced no tokens in the one-second host check.

Host CPU timings: PyTorch threads=4, ExecuTorch threadpool explicitly=4, three
warmups and ten measured iterations per fixed clip. AC power; the post-run OS
snapshot had no recorded thermal/performance warning. Desktop workload and
long-duration thermal behavior were not controlled.

| Fixed input | Median | p95 |
| --- | ---: | ---: |
| 1 second | 121.48 ms | 123.47 ms |
| 5 seconds | 388.21 ms | 402.01 ms |
| 10 seconds | 793.90 ms | 871.69 ms |

Timing includes padding, preprocessing, subsampling, encoder, autoregressive TDT
prediction/joint calls and host tensor copies, but excludes model loading and
final SentencePiece text decoding. These are full-utterance results, not
streaming first-word latency. Model load: 2.09 s; process peak RSS: 3,257,810,944
bytes (about 3.03 GiB). Historical 60/180-second crop batches took 5.13/13.62 s
of host ASR; silence and omitted spans were not processed. Do not compare those
times with the old complete simulator pipeline or claim phone performance.

Serialized inventory confirms XnnpackBackend delegates in every component;
portable operators remain. Delegate count is not a time or FLOP percentage.
No per-operator event trace or physical device measurement was performed.

Next integration gate: RNE dynamic schemas and asset lifecycle, source-matching
short-clip padding, VAD, stop/reset semantics, then an actual iOS simulator run.
FP32 artifacts are larger than the current quantized LFM asset. Quantization and
clean mode are separate, unvalidated candidates.


## Experimental dynamic INT8 follow-up

**Status: task-quality/size gates passed; experimental, not integration-ready.**
The strict full-encoder numerical checks below remain failed. Do not interpret
transcription agreement or simulator execution as a replacement for that check.
No quantization tolerance was relaxed. FP32 remains the verified source baseline.

Quantize only encoder linears with dynamic INT8 activations and per-channel INT8
weights, retaining other operations and four companion files in FP32. Source,
raw-mode decoding, tokenizer and audio policy are unchanged. No speech calibration
set is used; one zero-input execution initializes weight observers. TorchAO is
explicitly pinned to 0.18.0.

The combined greedy XNNPACK partitioner failed with a dependency-cycle error.
The successful candidate first lowers per-operation dynamic-quantization islands,
then merges FP32 regions. It contains 626 encoder delegate entries, all XNNPACK;
portable operations remain. Counts are not time/FLOP coverage. The converted EP
and comparison fixtures are retained beside the candidate for diagnosis. Never
package the intermediate `.pt2` or fixtures as application model assets.

From the repository root:

```sh
uv run --locked --project exporter_models/fluid_speech_pro python \
  exporter_models/fluid_speech_pro/quantize.py --output-dir /absolute/int8-output

# This check currently FAILS the strict numerical budget, and exits nonzero.
uv run --locked --project exporter_models/fluid_speech_pro python \
  exporter_models/fluid_speech_pro/validate_quantized.py \
  --pte /absolute/int8-output/fluid_speech_pro_encoder_xnnpack_int8dyn.pte \
  --reference /absolute/int8-output/encoder-quantized-reference.pt \
  --report /path/quantized-parity.json

uv run --locked --project exporter_models/fluid_speech_pro python \
  exporter_models/fluid_speech_pro/benchmark.py \
  --pte-dir /absolute/int8-output \
  --encoder-file fluid_speech_pro_encoder_xnnpack_int8dyn.pte \
  --tokenizer /path/tokenizer.model --fixture /path/fixture.json \
  --report /path/int8-benchmark.json
```

Predeclared task gates: at least 50% total PTE size reduction; WER no more than
one absolute percentage point above FP32 on either matched excerpt, and still
below historical LFM WER on both. Both gates passed:

| Measure | FP32 baseline | Mixed INT8 candidate |
| --- | ---: | ---: |
| Total PTE bytes | 2,558,481,192 | 974,770,088 (61.9% smaller) |
| Encoder bytes | 2,459,780,352 | 876,062,720 |
| 60s cohort WER | 16.38% | 16.38% |
| 180s cohort WER | 18.91% | 18.91% |
| Source token sequence matches | 102/102 | 101/102 |
| 1s host median / p95 | 121.48 / 123.47 ms | 83.44 / 91.12 ms |
| 5s host median / p95 | 388.21 / 402.01 ms | 222.20 / 225.66 ms |
| 10s host median / p95 | 793.90 / 871.69 ms | 386.12 / 389.88 ms |
| Process peak RSS | 3.03 GiB | 1.83 GiB |
| Model load | 2.09 s | 1.04 s |

The sole token-sequence change was capitalization: “you're doing the actual
planning.” became “You're doing the actual planning.” These are the same limited,
overlapping one-meeting crops described above. A subsequent FP32 run gave a
788.83 ms ten-second median, but short-clip timings/p95 varied substantially
(1s median 286.60 ms, 10s p95 1293.78 ms). Desktop memory/workload conditions are
uncontrolled; this is indicative host timing, not a stable speedup guarantee.

### Strict numerical checks and diagnosis

The predeclared converted-graph vs loaded-runtime budget was atol=rtol=1e-3.
Integer lengths, shapes, dtypes and finite-output checks passed. Floating outputs
failed at all three random embedding lengths 3/63/126: max absolute differences
0.00556/0.01574/0.01098. Every boundary result is retained, including failures.

A one-linear reproducer (`diagnose_dynamic_quant.py --output-dir /path/probe`)
shows a granularity difference: default PT2E uses `per_tensor_affine` activation
qparams, while XNNPACK linear uses `num_nonbatch_dims=1`, quantizing each token
row. One row agrees within 3.4e-8; multiple rows differ by up to 0.081 with the
global chooser. Evaluating the same converted linear per row reduces that error
to at most 1.91e-6, with zero violations of the existing budget.

`_quant_reference.py` evaluates activation Q/DQ row by row in the converted
encoder, retaining weights and other model operations. This narrows the mismatch,
but **does not resolve the full encoder**: max absolute errors at 3/63/126 frames
are 0.00289/0.01112/0.00852, still outside budget. Remaining accumulated/backend
numerical differences need localization. This evaluator changes no export or
native runtime code, and is a diagnostic rather than a passing reference claim.

```sh
uv run --locked --project exporter_models/fluid_speech_pro python \
  exporter_models/fluid_speech_pro/_quant_reference.py \
  --program /absolute/int8-output/encoder-quantized.pt2 \
  --fixtures /absolute/int8-output/encoder-quantized-reference.pt \
  --output /absolute/int8-output/encoder-rowwise-reference.pt
```

Use only locally generated trusted reference/EP files. Compare the resulting
reference with `validate_quantized.py`; it currently exits nonzero too.

### iOS simulator probe

The local probe uses the existing Gallery native release binary and its pinned
RNE source 4b7d6fd6fea10f9b3422121d6e439ccf88df93b2 (ExecuTorch 1.4.1), replacing
only the JavaScript bundle with a local validation entry. No RNE source changes,
new native build, Gallery tracked changes, model upload or PR integration.
Native dynamic-schema loading and all five components execute successfully.
Simulator numbers are individual observations, not physical-device performance.

Authoritative rerun: dedicated “Fluid Speech Validation” iPhone 17 Pro simulator,
iOS 26.5. All **14/14** calls matched host quantized token IDs: 1/5/10-second
speech, a sub-second crop, silence, all eight subsampling residue methods and a
repeated first clip after the other calls. No nonfinite encoder/joint outputs or
symbol-cap hits. Models/tensors are disposed after the probe. Maestro 2.12.1
asserted the pass result and captured the screen. A local probe is not a test of
meeting UI download/import, VAD, cancellation, backgrounding or storage deletion.

Dedicated simulator individual times: 507/324/709 ms for 1/5/10-second speech;
repeated 1s clip 101 ms. These include native model execution and JS tensor
copies, exclude final tokenizer decode, and are not warmup-separated timing
statistics. Do not use them as an iPhone benchmark. Source JSON tokenizer decoding
also matched SentencePiece on all 102 saved source sequences on the host.

### Candidate artifact hashes

| File | SHA-256 |
| --- | --- |
| `fluid_speech_pro_encoder_xnnpack_int8dyn.pte` | `b504890903c56c073f68fbf941c8df9b2f2f52a1724a31799d012c96d688318a` |
| `fluid_speech_pro_joint_xnnpack_fp32.pte` | `ac7d4958c4e051a82cfa92183084576aa856f1281ccaa1900a137205e013b8ea` |
| `fluid_speech_pro_predictor_xnnpack_fp32.pte` | `ef613e1548f6f77dbc9857dadf60344e265dc3759d1fd8f17e06ce23be3429f2` |
| `fluid_speech_pro_preprocess_xnnpack_fp32.pte` | `a6a3189cdaacfd88e608f55bfc6ea0bda21ea1b2795aae1b4b147913da109e68` |
| `fluid_speech_pro_subsample_xnnpack_fp32.pte` | `591a9b51a086ee969802a2f23b3b6cd6d4d1d4a255b87fbe04c8c781c9680561` |

Next gates: localize the remaining full-encoder numeric drift, add broader
held-out meeting/language evaluation, then validate actual meeting-pipeline
lifecycle and physical-device memory/latency before replacing LFM. Keep this
candidate experimental until those checks are resolved.
