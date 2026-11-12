# Nemotron 3 Diarization export

Candidate recipe, validation in progress. This does not establish device support.

Source: NVIDIA `Nemotron-3-Diarization`, checkpoint
`52a432672732425e4c19765b9fec9b5810e6f4ba`, OpenMDW 1.1 weight terms.
Code: pinned Transformers 5.20.0 (Apache-2.0). The preprocessing wrapper adapts
its `NemotronAsrStreamingFeatureExtractor` tensor operations. `THIRD_PARTY_LICENSE`
retains the code license; the source model's terms govern its weights.

Python 3.10, torch 2.14.0, ExecuTorch 1.4.1, Transformers 5.20.0. Transformers
5.21+ requires Python 3.11, so it is incompatible with this harness environment.
No global environment or RNE changes are required.

Acceptance budget, fixed before export: identical method/shape/dtype/metadata
contracts and finite task outputs. Features atol 1e-4 / rtol 1e-4, embeddings
atol 1e-4 / rtol 1e-4, speaker probabilities and pooled probabilities atol 1e-4 /
rtol 1e-3. Report maximum errors and threshold disagreement, including values
near 0.5. Compression must select the same embedding/probability values on
identical inputs; ties between identical silence frames are equivalent.
Validate each component and the complete streaming chain against the original
HF eager model on AMI meeting audio, short/partial buffers, repeated chunks and
cache overflow. The official eight-voice sample is a separate synthetic test,
not a real-meeting benchmark. Do not claim DER without a proper reference.

The fixed preprocessing capacity is 10 s / 160000 mono samples at 16 kHz;
true-length masking permits partial buffers. The dynamic embed method takes
zero-padded mel frames in multiples of eight. The dynamic encode method returns
10 ms speaker probabilities and masked pooled 80 ms cache probabilities.
Compression uses the original HF speaker-cache policy, including learned silence
embeddings, speaker grouping and confidence boosts. The app owns only FIFO/cache
counters and bounded latent tensors. This avoids the older export's 1000-frame
padding penalty and its simplified cache policy.

Commands, artifact hashes, parity and target results will be recorded below.

Attempt 01: preprocessing, embed and encode capture passed. Compression failed
at torch.export because the original in-place tail score boost specialized
`cache_frames - 264 != 1`, excluding the valid 265-frame boundary. Replace only
that view mutation with a functional per-frame boost mask (marked EXPORT PATCH),
retain the complete domain 265–800, and compare cache selections with the
unmodified source at 265/266/intermediate/max before retrying capture.

Attempt 02: all torch.export captures passed, including 265-frame compression.
The companion-schema validator rejected bool as a public tensor dtype (the
canonical schema allows float32/uint8/int32/int64). Change the exported validity
mask to int64 (0/1), convert it to bool inside the encoder, and preserve its
original boolean attention semantics. No runtime or schema validator bypass.

Attempt 03: capture/lowering passed, serialization failed because FlatBuffers'
JSON parser rejects Infinity from cache ranking sentinels. Map only the internal
positive/negative infinity ranking sentinels to float32 max/min. Actual log scores
are bounded by the original 0.25 probability clamp; score boosts are too small to
change either extreme, so ranking and rejection behavior remain equivalent.
Do not change model logits, probability thresholds or parity budgets. Check
identical selected cache values at all boundary cases before another export.

Attempt 04 produced 397,708,832 bytes. Host ET1.4.1 loaded preprocessing/embed/
encode but rejected compress: missing `aten::sort.values`. The pinned RNE native
binary also exposes topk but no sort kernel. Replace the ascending sort of all
264 selected indices with `topk(k=264, largest=False, sorted=True)`. This sorts
exactly the same full vector, including duplicate values, using the registered
operator; no runtime/library change or cache approximation. Recheck original
cache outputs before export attempt05.

Attempt 05: **host parity passed** on all four loaded methods, including min/
intermediate/max shapes, silent/saturated cache cases and six consecutive
10-second AMI IB4001 buffers (210–270s). Two cache compressions were exercised.
Maximum meeting-stream speaker probability absolute error: 2.2649765e-06.
No values exceeded the predeclared tolerances. Compression selections on identical
inputs matched the original eager policy exactly.

The NVIDIA eight-voice synthetic sample (97.6s, 10 buffers, four compressions)
also passed original-eager parity. All eight output channels had above-threshold
activity. Maximum streaming probability absolute error: 3.7103891e-05.
This verifies export fidelity, not speaker identity accuracy or DER.

Artifact: `nemotron3_diarization_xnnpack_fp32.pte`, 397708832 bytes.
SHA256: `e44ba5ebf7f9f6c92c363f5b1ee010186e3f0fe54c4db3e38f31bfef14a605e3`. Native iOS simulator execution passed in Gallery (ET1.4.1,
iPhone 17 Pro/iOS26.5 on M1 Pro): real AMI 210–270s with LFM3 Audio completed in
14.415s; extending to 210–390s completed in 48.772s, exercising repeated cache
compression. Approximate word error rates were 18.1% (32/177) and 21.7% (108/497)
respectively. Those measure the combined ASR/cropping pipeline, not diarization
accuracy. The longer excerpt has a brief spurious Speaker 6 in a four-person
meeting; DER was not measured. The native synthetic eight-voice input completed
in 5.304s. These are simulator timings, not physical phone performance.
No physical microphone or Android checks were performed.

```sh
uv run --locked --project exporter_models/nemotron3_diarization exporter export nemotron3_diarization --backend xnnpack --output-dir /absolute/ignored/output
uv run --locked --project exporter_models/nemotron3_diarization python exporter_models/nemotron3_diarization/validate.py --pte /absolute/ignored/output/nemotron3_diarization_xnnpack_fp32.pte --audio /path/IB4001.Mix-Headset.wav --start 210 --chunks 6 --report /absolute/ignored/ami-parity.json
uv run --locked --project exporter_models/nemotron3_diarization python exporter_models/nemotron3_diarization/validate.py --pte /absolute/ignored/output/nemotron3_diarization_xnnpack_fp32.pte --audio /path/nemotron-eight-voices.wav --chunks 10 --skip-boundaries --report /absolute/ignored/eight-parity.json
```

Validation audio sources:
- Real AMI meeting: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IB4001/audio/IB4001.Mix-Headset.wav
- Synthetic eight voices: https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/52a432672732425e4c19765b9fec9b5810e6f4ba/nemotron3_tts_8_open_voices.mp4 (extract mono16k WAV).

No weights/audio are committed or uploaded. The Gallery PoC can import this local
artifact. No RNE source modifications are needed.
