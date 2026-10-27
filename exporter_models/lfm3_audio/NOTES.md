# lfm3_audio

LiquidAI/LFM3-Audio-2B, pinned at revision
`9d49701c372d6011d900f6abb40c6b296fb706dd`, exported for **audio in, text
out** (ASR, speech translation to text, spoken questions). The text decoder is
the LFM3-1.7B architecture (`lfm3_text/NOTES.md`; code copied) with its own
weights under `lfm.model.`. Speech output (depth decoder, 12 codebooks,
LiquidCodec) is not exported: RNE has no audio-generating LLM runner.

## Contract

```
audio_encoder   f32[1, 400 + 160k] 16 kHz mono, k <= 2997 (30 s) -> f32[1, T, 2048], T = ceil((k + 1) / 8)
token_embedding i64[1, 1..256]                                   -> f32[1, S, 2048]
text_decoder    f32[1, 1..256, 2048], i64[S]                     -> f32[1, 128000]
```

The caller right-pads the clip the way the feature extractor does (to at
least 400 samples, then up to the next 160-sample hop); `get_model_schema`
declares the waveform dim as a range with step 160. One audio token per 8
mel frames (12.5 tokens/s). In the prompt, a clip is a bare run of T
`<|audio|>` tokens (no start/end markers). Context 8192. The constant method
`get_audio_frames_per_token` (8) lets the runner size the encoder output
without hardcoding the downsampler stride.

`text_decoder` adds a constant mask that sets `<|audio|>`, `<|audio_out|>`,
`<|audio_start|>` and `<|mixed_start|>` logits to -inf, which is what
upstream `generate(suppress_mixed=True, suppress_audio_out=True)` does for
text-only output, so an on-device sampler cannot switch into speech mode.

## Audio encoder (`_lfm3/audio.py`)

The log-mel front end is in the graph: a stride-160 conv1d against a
Hann-windowed DFT basis (no `aten.stft`), |X|^2, Slaney mel filters (from
`transformers.audio_utils.mel_filter_bank`, the extractor's own call), log10
clamped at -12, then (x + 4) / 4. Then three stride-2 convs, a 17-layer LFM2
trunk (short convs, attention limited to the last 32 tokens, causal) and the
LayerNorm/GELU projector. Against upstream `get_audio_features` (bf16
reference): per-token cosine >= 0.99985 on the 18.4 s Harvard-sentences clip
(230 tokens). Unquantized, like the VL vision tower.

## Results

Eager vendored pipeline vs HF (`validate.py --eager`): 48/48 teacher-forced
greedy tokens, transcript "The stale smell of old beer lingers. It takes heat
to bring out the odor. ...". Exported variants: see the MR.

Vulkan: same blocker as `lfm3_text` (pytorch/executorch#23539).
