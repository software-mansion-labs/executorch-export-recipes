# smollm2

## Config is vendored, not fetched

Same two repo-wide errors as the other families: `get_n_layers: null` (real
counts are 30 / 32 / 24) and a `forward` sequence dim of 2048 where the `.pte`
declares 2047. Corrected here; everything else matched.

## The published 8da4w files for 135m and 360m are not really quantized

`smollm2_135m_xnnpack_8da4w.pte` is 0.56 GB against 0.27 GB for bf16, and 360m
is 1.36 GB against 0.72 GB. The metadata and IO contract of both are correct, so
this is not a mislabelled upload. The cause is a silent executorch behaviour:

`qmode=8da4w` with `group_size` unset falls back to **128**, and the quantizer's
filter then only takes a Linear when `in_features % group_size == 0`. At
dim 576 (135m) and 960 (960 % 128 = 64) that skips every attention projection
and the two MLP layers that read from `dim`; only the one Linear reading from
`hidden_dim` survives. The result is a nearly-unquantized fp32 model.

That is also why the old sweep ranked `8da4w_default` best for exactly these two
sizes: it was scoring a model that had barely been quantized, so it won the
perplexity/KL-divergence axes by construction.

`_params.py` therefore drops the sweep's winner for these two sizes and takes
the best *genuinely quantized* config its own `results.csv` measured:

| size | recipe | ppl | size |
|---|---|---|---|
| 135m | bf16 | 12.40 | 258 MB |
| 135m | `8da4w_default` (the sweep's pick) | 14.00 | **535 MB** |
| 135m | `gs32 + emb 8,0` (**used here**) | 16.61 | 101 MB |
| 135m | `gs64 + emb 8,0` | 19.10 | 97 MB |
| 360m | `8da4w_default` (the sweep's pick) | 9.84 | **1301 MB** |
| 360m | `gs32 + emb 8,0` (**used here**) | 11.29 | 241 MB |
| 360m | `gs64 + emb 8,0` | 12.72 | 230 MB |

Note the sweep recorded the 535 MB / 1301 MB sizes all along -- larger than the
bf16 they were meant to compress -- so the trap was visible in its own output.

`group_size 32` beats 64 by 13% / 11% perplexity for a few MB, and both divide
576 and 960. 1_7b (dim 2048, hidden 8192) was never affected and keeps the
sweep's `gs64 + emb 8,64`; there `gs32` diverges outright (ppl 4e9), which is
why the group size is chosen per size rather than repo-wide.

There is a real quality cost to quantizing these two for the first time: 135m
goes from ppl 14.00 (barely quantized, 535 MB) to 16.61 (101 MB). That is the
trade the sweep's Pareto was meant to expose; if the 135m/360m quantized
variants are not worth shipping at that quality, the honest alternative is to
drop them and ship bf16 only for those sizes.

## Published artifacts

The six shipped `.pte` all match this config's IO and runner constants:

```bash
uv run --project exporter_models/smollm2 python tools/probe_published_pte.py smollm2 --size 135m
```

But 135m and 360m `_8da4w.pte` were built with the trap above, so a re-export is
needed for the quantized variant to actually be quantized. bf16 is unaffected.

## Measured on device (Galaxy S26 Ultra, SM-S948B, Android 16)

Four builds per size, executorch's `llama_main` (arm64, ET 1.4.1 @226b471),
greedy, `--cpu_threads=4`:

| 135m | file | decode | RSS | "reverse a string" |
|---|---|---|---|---|
| bf16 (the original) | 0.270 GB | 14 tok/s | 433 MiB | correct |
| published `_8da4w` | 0.561 GB | 93 tok/s | 692 MiB | correct |
| **int8 per-channel (shipped)** | **0.166 GB** | **248 tok/s** | **316 MiB** | **correct** |
| int4 gs32 | 0.106 GB | 335 tok/s | 250 MiB | nonsense `re.sub` code |

| 360m | file | decode | RSS | "explain gravity to a six year old" |
|---|---|---|---|---|
| bf16 (the original) | 0.725 GB | 11 tok/s | 935 MiB | correct |
| published `_8da4w` | 1.364 GB | 37 tok/s | 1555 MiB | correct |
| **int8 per-channel (shipped)** | **0.412 GB** | **106 tok/s** | **649 MiB** | **correct** |
| int4 gs32 | 0.253 GB | 152 tok/s | 481 MiB | refuses: "I'm not a six-year-old..." |

int8 is the variant worth shipping at both sizes: smaller than the bf16 it sits
beside, far smaller than the file it replaces, an order of magnitude faster than
bf16, and it tracks the bf16 answers. 4-bit buys another ~40% size and speed and
loses the model.

## The published filename changes

Because the variant is genuinely int8 now, its `precision` is `8da8w` and the
file is `smollm2_{135m,360m}_xnnpack_8da8w.pte`, not `_8da4w.pte`. Whoever
republishes has to add the new file and update the consuming URL in
react-native-executorch's model registry; the old `_8da4w.pte` names describe a
build we are deliberately not producing any more. 1_7b is untouched and keeps
`smollm2_1_7b_xnnpack_8da4w.pte`.

## 1_7b: the 4-bit recipe is worse than it first looked

1_7b has the largest measured degradation of the three sizes (ppl 6.50 -> 11.99,
1.8x, against 1.3x at 135m and 1.25x at 360m), so it was exported as int8 and
compared on device against the published 4-bit build.

The first comparison used three prompts and favoured keeping int4. That sample
was too small: it happened to miss int4's characteristic failure modes. Widened
to ten prompts, greedy decoding (`--temperature=0`), 110-token budget:

| prompt | int4 | int8 |
|---|---|---|
| capital of France | ok | ok |
| reverse a string | ok | ok |
| explain gravity | **wrong physics** | ok |
| 3pm + 90 minutes | wrong (3pm) | wrong (4pm) |
| three planets | glitch ("700-year-old dust") | glitch ("fourth planet from the man") |
| why ice floats | ok | stop-token derail, see below |
| write a haiku | **repetition loop to the cap** | ok |
| list vs tuple | **states it backwards** | ok |
| console a friend | ok (one invalid UTF-8 byte) | ok |
| explain primes | **loops, muddled** | ok |

int4 fails 4 of 10; int8 fails 0 of 10 outright. Neither can do the arithmetic.

int4's failures are the classic signature of over-quantization: degenerate
repetition ("Rhythmic, Rhythmic, Rhythmic..." to the token cap on the haiku, and
"But if you had a number, say..." on primes), plus a confidently reversed fact
-- it tells the user that lists "can be modified, but not directly, while tuples
can", which is exactly backwards and is the kind of thing a coding assistant
must not say.

For the record, two claims in the earlier version of this section were wrong.
int4's gravity answer does **not** run to the token cap -- that was an artefact
of an 80-token budget; at 110 tokens it terminates on EOS. And "the device did
not confirm it" no longer holds: with ten prompts the device confirms the
perplexity signal.

### The ice-float derail is a stop-token gap, not quantization

int8 on "why does ice float" starts correctly, emits `<|endoftext|>` mid
sentence, and then continues into unrelated lowercase pretraining text about the
invasion of Iraq. It reproduces byte-for-byte across runs (greedy decoding).

The cause is the config, not the weights: in the smollm2 tokenizer token 0 is
`<|endoftext|>` and token 2 is `<|im_end|>`, and every smollm2 config declares
`"get_eos_ids": [2]`. When the model emits token 0 the runner does not recognise
a stop and keeps generating. Upstream declares `eos_token_id = 2` too, so the
config is faithful to the reference -- but the gap is latent for **all** smollm2
sizes and both precisions, and int4 simply did not happen to emit token 0 on
these ten prompts.

`get_eos_ids` is a list precisely so a model can carry more than one stop token
(hammer2_1 in this MR carries `[151645, 151643]`). Declaring `[2, 0]` for
smollm2 would turn this from "generates unrelated content" into "stops early".
That is a config change, not a re-export, so it is cheap and independent of the
precision decision below.

### bf16 is not a candidate

The tiebreaker finally ran (the file is 3.42 GB; USB kept dropping the single
push, so it went over in 256 MB chunks and was verified on device against the
published sha256):

| 1_7b | file | decode | RSS |
|---|---|---|---|
| int4 | 1.018 GB | 40-46 tok/s | 1955 MiB |
| int8 per-channel | 1.817 GB | 25-27 tok/s | 2104-2737 MiB |
| bf16 | 3.424 GB | **1.5-1.9 tok/s** | 3702 MiB |

bf16 answers gravity correctly and cleanly, which settles the question the
tiebreaker was for: int4's wrong answer **is** caused by quantization, not by
the model or the prompt. But bf16 decodes ~25x slower than int4 -- it falls off
a memory-bandwidth cliff at 3.4 GB -- so it is a reference point, not a
shippable variant.

### Recommendation (not yet applied)

Switch 1_7b to int8 per-channel, at +78% file size (1.018 -> 1.817 GB) and -40%
decode (40-46 -> 25-27 tok/s), and add token 0 to `get_eos_ids` for all three
sizes. The recipe in `_params.py` is deliberately left on 4-bit until that is
agreed, because changing it means a re-export and a republish.
