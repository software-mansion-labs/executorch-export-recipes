# Fluid Speech Pro: full FP32 XNNPACK export

Validated on 2026-11-16 on an Apple M1 Pro, 16 GiB, macOS host CPU.
This is a host-verified export and benchmark, not an iOS/Android integration.
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
- These files do not embed RNE-specific `get_model_schema` constants. Native
  integration still needs the explicit dynamic schema/asset handoff and a check
  against the PR-pinned shipping runtime; host loading is not that validation.

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
