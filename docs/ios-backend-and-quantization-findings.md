# Quantization schema sweep (host-side), ET 1.4.1 fork `1.4.1+226b471`

Goal: per model, find which {backend x quantization} artifacts are worth shipping,
and drop the dominated ones. Host-only: this decides **which candidates exist**,
not the final speed ranking, which still needs device measurement.

Method note that applies to every run below: input tensors must be `.contiguous()`.
A dense-but-non-contiguous input makes `Method.execute()` read storage in the wrong
order and return deterministic garbage. Every sweep here starts with a CONTROL row
(unquantized, known-good backend) and is only trusted if the control lands on the
reference.

---

## 1. efficientnet_v2_s (conv CV) @ 384x384, 24 VOC images

| backend | scheme | MB | top1 | cos | host_ms |
|---|---|---:|---:|---:|---:|
| xnnpack | fp32 (control) | 85.7 | 1.00 | 1.000000 | 50.9 |
| coreml | fp32 | 86.5 | 1.00 | 1.000000 | 8.8 |
| coreml | **fp16** | 43.9 | 0.96 | 0.999619 | 2.4 |
| coreml | **fp16 + int8/per_channel** | 22.1 | 0.92 | 0.998859 | 2.4 |
| coreml | fp16 + int8/per_tensor | 21.9 | 0.71 | 0.980532 | 2.4 |
| coreml | fp16 + int8/per_channel/linear | 22.3 | 0.92 | 0.998937 | 2.4 |
| coreml | fp16 + uint8/per_channel | 22.2 | 0.92 | 0.998871 | 2.4 |
| coreml | fp16 + int4/per_channel | 11.5 | 0.04 | 0.196040 | 2.4 |
| coreml | fp16 + int4/per_block(32) | 14.0 | 0.71 | 0.883359 | 10.5 |
| coreml | fp16 + uint4/per_block(32) | 14.1 | 0.71 | 0.883358 | 10.9 |
| coreml | fp16 + int8/per_block(32) | 24.2 | 1.00 | 0.999745 | 7.9 |
| mlx | fp32 / fp16 / 4w / 8w / nvfp4 | 86.7 / 43.5 / 82.4 / 82.9 / 86.7 | >=0.96 | >=0.9972 | ~25 |

`host_ms` is macOS, not iPhone. Compare within a backend only.

**MLX quantization is structurally useless for conv CV models.** `quantize_model_`
rewrites only `nn.Linear`/embeddings. EfficientNet-V2-S has exactly one Linear
(`classifier.1`), **1,281,000 of 21,458,488 params = 6.0%**. `4w` buys 5% size for
cos 0.9973. `nvfp4` is a literal no-op here: it logs `Skipping classifier.1: weight
shape [1000,1280] incompatible with nvfp4 (group_size=16)` and emits a byte-identical
artifact at cos exactly 1.0. Generalizes to every conv-dominated model we ship.

**CoreML Pareto set is two entries**; everything else is dominated. `per_tensor` is
same-size but much worse than `per_channel`; `uint8` and `mode=linear` are
near-duplicates of symmetric int8; `int4/per_channel` is destroyed (cos 0.196);
`int4|uint4/per_block(32)` lose on BOTH axes (cos 0.883 and ~4x slower);
`int8/per_block(32)` has the best quantized numerics but is bigger AND 3x slower
than per_channel. Survivors: **fp16 (43.9 MB)** and **fp16+int8/per_channel
(22.1 MB)** at equal host latency.

Open risk: the per_block slowdown may not hold on an iPhone ANE. If it fuses there,
`int8/per_block(32)` becomes the best quantized option.

---

## 2. lfm2.5-embedding-350m (MLX) -- Linear-dominated, MLX-only on iOS

Chosen because it is the largest Linear-dominated model we ship, Core ML is
de-declared for it (NOTES 0x23 load failure), and `mlx/config.json` shipped ONE
variant (4w/64, `default:true`) that NOTES admitted was "not yet verified on iOS".

Eval: 30 query/doc pairs in 10 topical clusters of 3, so each query has two strong
same-topic distractors. An earlier 12-pair set saturated at r@1 1.00 for every scheme
and was discarded as non-discriminating.

| scheme | MB | cos_ref | spear | r@1 | MRR | mean_rank |
|---|---:|---:|---:|---:|---:|---:|
| fp32 eager (reference) | - | 1.000000 | 1.0000 | 0.967 | 0.9833 | 1.033 |
| bf16 unquantized (control) | 709.0 | 0.999940 | 0.9999 | 0.967 | 0.9833 | 1.033 |
| **4w/64 lin + 4w/64 emb [SHIPPED]** | **188.5** | 0.968356 | 0.9721 | 0.967 | 0.9833 | 1.033 |
| 4w/32 lin + 4w/32 emb | 199.6 | 0.974859 | 0.9728 | 0.967 | 0.9833 | 1.033 |
| 4w/128 lin + 4w/128 emb | 183.0 | 0.960429 | 0.9648 | 0.967 | 0.9833 | 1.033 |
| 8w lin + 8w emb | 360.2 | 0.999587 | 0.9995 | 0.967 | 0.9833 | 1.033 |
| 8w lin + 4w/64 emb | 327.7 | 0.998517 | 0.9970 | 0.967 | 0.9833 | 1.033 |
| 4w/64 lin + 8w emb | 221.0 | 0.969608 | 0.9729 | 0.967 | 0.9833 | 1.033 |
| 4w/64 + 4w/64, SKIP CONV | 250.1 | 0.969640 | 0.9709 | 0.967 | 0.9833 | 1.033 |
| 8w + 8w, SKIP CONV | 401.5 | 0.999601 | 0.9994 | 0.967 | 0.9833 | 1.033 |
| nvfp4 lin + nvfp4 emb | 199.6 | 0.929460 | 0.9215 | **0.900** | **0.9444** | **1.133** |

**The shipped 4w/64 is vindicated** -- 188.5 MB (3.8x off bf16) with retrieval exactly
at the fp32 reference. No config change needed.

**Reject nvfp4.** Only scheme that moves retrieval, and it is *larger* than 4w/64
while doing it. Strictly dominated. Note this contradicts the EfficientNet reading,
where nvfp4 looked harmless purely because it silently skipped the sole Linear --
a reminder that a no-op can masquerade as a good result.

**Do not port `_quantize_8da4w_skip_conv` to MLX.** The XNNPACK recipe skips the
short-conv projections because quantizing them "scrambles embeddings on ARM XNNPACK".
On MLX that skip costs +62 MB and buys cos 0.9696 vs 0.9684, i.e. noise. Keep the
workaround XNNPACK-scoped.

**MLX preserves the dynamic seq dim.** Token lengths 6..30 all executed against a
`{1..511}` export. NOTES records XNNPACK specializing that dim to its max on this
build; MLX does not. The local static-seq config deviation is an XNNPACK problem only.

Caveats: no latency measured, so by the "keep two only if the smaller one is slower"
rule the evidence supports shipping 4w/64 ALONE; and 30 pairs only resolves gross
damage (it separated nvfp4 and nothing else), so ordering inside the 4w/8w family
rests on cos_ref alone and would need a real MTEB run to claim more.

---

## 4. lfm2.5-colbert-350m (MLX) -- late interaction, same backbone

Same 30-pair hard corpus, but scored with **MaxSim** late interaction
(`sum_i max_j q_i.d_j`), so quantization error is not averaged away by a single
pooled vector. `cos_ref` is the mean PER-TOKEN cosine vs the fp32 eager reference.

### 4a. Blocking bug found first: the MLX artifact never loaded

`mlx/export.py` lowered with neither `transform_passes=get_default_passes()` nor
`_skip_dim_order=True`. Every artifact it produced -- including unquantized bf16 --
failed with `Failed to load method forward, error: 0x23`. `model.json` declares
`mlx` as a supported backend and NOTES called it "structural-only", so this shipped
broken. Full truth table (`0x23` unless noted):

| get_default_passes | _skip_dim_order | dict call form | result |
|:--:|:--:|:--:|---|
| no | no | no | FAIL (as-shipped) |
| no | yes | no | FAIL |
| yes | no | no | FAIL |
| no | no | yes | FAIL |
| yes | no | yes | FAIL |
| no | yes | yes | FAIL |
| **yes** | **yes** | yes | **OK** |
| **yes** | **yes** | no | **OK** |

Both are required, and only in conjunction; the dict vs positional call form is
irrelevant. Fixed in `mlx/export.py`; the sibling embedding exporter always had both,
which is why it worked. Note this `0x23` is the SAME code NOTES reports for the Core ML
builds of both lfm2.5 models -- so that Core ML finding is worth re-testing against a
lowering-config cause rather than the assumed constant-segment/size cause.

### 4b. Sweep (after the fix)

| scheme | MB | cos_ref | spear | r@1 | MRR |
|---|---:|---:|---:|---:|---:|
| fp32 eager (reference) | - | 1.000000 | 1.0000 | 1.00 | 1.000 |
| bf16 unquantized (control) | 707.0 | 0.999983 | 0.9996 | 1.00 | 1.000 |
| 4w/64 lin, no qembedding [SHIPPED] | 284.9 | 0.993627 | 0.9928 | 1.00 | 1.000 |
| **4w/64 lin + 4w/64 emb** | **188.0** | 0.993265 | 0.9947 | 1.00 | 1.000 |
| 4w/32 lin + 4w/32 emb | 199.0 | 0.994328 | 0.9978 | 1.00 | 1.000 |
| 4w/128 lin + 4w/128 emb | 182.5 | 0.991628 | 0.9922 | 1.00 | 1.000 |
| 8w lin + 8w emb | 359.2 | 0.999908 | 0.9996 | 1.00 | 1.000 |
| 8w lin + 4w/64 emb | 327.2 | 0.999493 | 0.9994 | 1.00 | 1.000 |
| 4w/64 + 4w/64, SKIP CONV | 249.6 | 0.994816 | 0.9952 | 1.00 | 1.000 |
| nvfp4 lin + nvfp4 emb | 293.8 | 0.982757 | 0.9658 | 1.00 | 1.000 |

**Actionable: the exporter leaves 97 MB on the table.** It sets `qlinear_config` only
and never `qembedding_config`. Adding 4w/64 embedding quantization takes the artifact
from 284.9 MB to **188.0 MB (-34%)** at identical quality -- cos 0.99327 vs 0.99363 and
spear 0.9947 vs 0.9928 (marginally *better*). The embedding sibling already quantizes
its embeddings. Needs a publish decision since it changes the HF artifact.

**Late interaction is far more quantization-tolerant than pooled embedding.** At 4w
ColBERT holds cos 0.993 where the pooled embedding model drops to 0.968. MaxSim over
per-token vectors absorbs weight noise that a single pooled vector does not.

**r@1/MRR saturate completely at 1.00 for every scheme, including nvfp4.** MaxSim is
too strong for a 30-pair corpus, so for ColBERT the retrieval metrics carry no signal
at all and only cos_ref/spear discriminate. Do not read "r@1 1.00" as validation here.

**Cross-model confirmations** (both lfm2.5 models agree): reject `nvfp4` -- worst
cos_ref and spear of any working scheme on both, while not being the smallest. And
SKIP CONV is not worth porting to MLX -- +62 MB on both models for cos deltas inside
noise.

---

## 5. DEVICE results (iPhone 16, Release build, 3 warmup + 15 timed)

Measured with a model-agnostic benchmark screen in apps/computer-vision that
allocates inputs from each model's own `schema.forward`, reports
`model.backends` (so a silent portable fallback cannot masquerade as a slow
delegate), and checks output finiteness. Every row below delegated correctly
and returned finite output.

### 5a. efficientnet_v2_s @384 (conv CV)

| artifact | MB | median ms | min | p90 | backend |
|---|---:|---:|---:|---:|---|
| XNNPACK fp32 | 85.7 | 125.5 | 117.4 | 127.9 | Xnnpack |
| CoreML fp32 | 86.5 | 16.9 | 16.7 | 18.1 | CoreML |
| **CoreML fp16** | 43.9 | **2.4** | 2.3 | 3.1 | CoreML |
| **CoreML int8/per_channel** | **22.1** | **2.4** | 2.3 | 3.0 | CoreML |
| CoreML int8/per_block(32) | 24.2 | 15.2 | 13.6 | 17.9 | CoreML |
| MLX fp32 | 86.7 | 45.0 | 44.8 | 45.2 | MLX |
| MLX fp16 | 43.5 | 47.0 | 45.9 | 47.1 | MLX |

- **per_block is rejected on device too.** The host flagged it 3.3x slow and I
  logged that as possibly a Mac-ANE artifact that might invert on iPhone. It got
  WORSE: 6.3x (15.2 vs 2.4 ms). Open risk closed, in the reject direction.
- **fp16 and int8/per_channel tie at exactly 2.4 ms.** No speed/memory tradeoff
  exists; the axis is size vs accuracy only. Ship int8/per_channel (half the
  size, same speed); keep fp16 only as an accuracy option. The top1 gap behind
  it (0.92 vs 0.96) is one image out of 24 -- thin evidence.
- **MLX is not competitive for conv CV**: 19x slower than CoreML, and fp16 buys
  it nothing (47 vs 45 ms, marginally worse).

### 5b. lfm2.5 pair @128 tokens (Linear-heavy text)

| artifact | MB | median ms | min | p90 | backend |
|---|---:|---:|---:|---:|---|
| **emb MLX 4w/64 [SHIPPED]** | **188.5** | **74.2** | 73.8 | 74.4 | MLX |
| emb MLX 8w + 4w/64 emb | 327.7 | 74.3 | 74.2 | 74.4 | MLX |
| emb XNNPACK 8da4w [current default] | 574.8 | 227.3 | 215.9 | 232.0 | Xnnpack |
| cb MLX 4w/64, no qembedding [SHIPPED] | 284.9 | 60.9 | 60.3 | 60.9 | MLX |
| **cb MLX 4w/64 + 4w/64 emb** | **188.0** | 61.3 | 60.0 | 61.4 | MLX |
| cb XNNPACK 8da4w [current default] | 426.0 | 163.6 | 155.5 | 166.6 | Xnnpack |

- **Weight quantization does not affect latency at all here.** 4w vs 8w linear is
  74.2 vs 74.3 ms. MLX's quantized kernels land on the same compute path, so
  quantization buys size only. This collapses the "keep more than one variant"
  rule: with no speed axis, ship the SMALLEST variant that holds quality.
- **MLX beats XNNPACK on BOTH axes**: embedding 3.1x faster and 3.1x smaller,
  ColBERT 2.7x faster and 2.3x smaller. MLX should be the iOS default for this
  pair, not XNNPACK.
- **ColBERT: adopt 4w/64 + 4w/64 emb.** -97 MB (-34%), quality identical, and
  +0.4 ms (0.7%). The 0.4 ms is real, not noise -- it holds across median, mean,
  min and p90 -- but it is a good trade. Applied to `mlx/export.py`.
- **Embedding: keep 4w/64.** 8w costs +139 MB for zero speed gain.

### 5c. The cross-cutting rule

MLX is wrong for conv CV (19x behind Core ML) and right for Linear-heavy text
(3x ahead of XNNPACK). "Fastest iOS backend" must be decided per model CLASS,
not once globally. Consistent with MLX being matmul-oriented.

### 5d. Host vs device fidelity

Ordering was preserved, magnitudes were not: XNNPACK came out 2.5x worse on
device than its host ratio implied, MLX ~1.9x worse. Host can pre-rank
candidates; it cannot price them. The funnel (host filters, device decides)
holds.

### 5e. Gaps in the device numbers

- **Cold start is NOT measured.** `loadModel` returned in 0-1 ms for 86 MB
  artifacts, i.e. loading is lazy/mmap and the 3 warmup runs absorb Core ML's
  compile step. First-run cost can be seconds for Core ML and is a real shipping
  concern; needs separate measurement.
- **No peak memory.** RNE exposes no used-memory API. Matters most for the 350M
  models, where RSS plausibly outranks the 97 MB file-size delta.

### 5f. Side finding: the static-seq config deviation may be removable

Both lfm2.5 models ship a LOCAL static-seq `config.json`, justified in NOTES by
XNNPACK lowering baking the dynamic seq dim to its max on ET 1.3.1. On ET 1.4.1
I probed all four artifact families and every one accepts BOTH 128 and max
length. NOTES states the revert condition as "once an ET build preserves the
dynamic seq through XNNPACK lowering" -- that condition now appears met.

---

## 6. Core ML compute_unit (iPhone 16, fp16, 3 warmup + 15 timed, median)

No Core ML artifact we ship pins `compute_unit`, so every one uses the default
`ALL`. ssdlite320 suggested that default was wrong. Sweeping a multi-resolution
model plus a second architecture shows **ssdlite is the outlier, not the rule.**

| model | ALL (default) | CPU_ONLY | CPU_AND_NE | CPU_AND_GPU |
|---|---:|---:|---:|---:|
| yolo26-s @384 | **2.0** | 8.3 | 2.0 | aborts |
| yolo26-s @512 | **2.7** | 12.0 | 2.6 | aborts |
| yolo26-s @640 | **3.8** | 20.0 | 3.7 | aborts |
| efficientnet @384 | **2.4** | 17.3 | 2.4 | 18.9 |
| ssdlite320 @320 | 9.5 | **7.2** | 9.7 | 11.8 |

**Do not blanket-pin CPU_ONLY.** The ANE wins by 4.2x to 7.2x on yolo26 and
efficientnet. Generalising from ssdlite would have caused a 4-7x regression
across most of the Core ML set. The default is correct almost everywhere.

**Input size scales the ANE advantage but never flips it**: 4.2x at 384, 4.4x at
512, 5.3x at 640. More conv work per inference amortises ANE dispatch overhead.
So ssdlite is not explained by "small input" -- yolo26 @384 still prefers ANE
strongly.

**ssdlite's likely cause is graph fragmentation, not scale.** Its lowering is 2
delegate calls plus 7 portable ops (it skips `aten.select_scatter.default`), so
the ANE path is repeatedly interrupted and transition cost exceeds the saving.
yolo26 and efficientnet run larger uninterrupted delegated regions.

**`ALL` == `CPU_AND_NE` everywhere measured** (2.0/2.0, 2.7/2.6, 3.8/3.7,
2.4/2.4), so the default already means "prefer ANE" and pinning it buys nothing.

**The Core ML GPU is never worth using for these graphs.** It ABORTS yolo26 at
every resolution with `MPSGraphExecutable.mm:5070: failed assertion 'Error: MLIR
pass manager failed'` -- an assertion, so SIGABRT, uncatchable, and it kills the
app. On efficientnet it merely loses (18.9 ms, worse than CPU_ONLY, widest
spread of the study: 17.2 to 22.9).

Note this corrected an existing repo note: the yolo26 exporter attributed that
abort to fp32 compute_precision. Precision is not the cause. fp16 with
compute_unit pinned to CPU_AND_GPU hits the identical assertion at 384/512/640.
The trigger is the GPU path itself; fp32 merely routed there by default.

### 6a. Can ssdlite's graph fragmentation be reduced?

Investigated, and the answer from our side is no.

The "7 portable ops" figure overstates the fragmentation. Five are `getitem`
(tuple indexing) and one is `alloc`; both are graph plumbing with no compute.
There is exactly **one** real portable op, `aten.select_scatter`, and it splits
the delegated region into 2 segments.

It cannot be removed here:

- Dropping `skip_ops_for_coreml_delegation=["aten.select_scatter.default"]` makes
  the export fail outright: `ValueError: In op, of type slice_update, named
  aten_select_scatter_default`. Core ML cannot compile it.
- The op is **not in our model graph**. `torch.export` produces zero scatter
  nodes, and so does `to_edge` (1520 nodes, zero scatter), and so does
  `to_edge_transform_and_lower` with no partitioner. It appears only once
  `CoreMLPartitioner` runs, so it is introduced by the Core ML backend's own
  transform passes and then rejected by Core ML itself.

So this is an upstream ExecuTorch/coremltools issue, not something a change to
the ssdlite wrapper can fix. Pinning `compute_unit=CPU_ONLY` remains the only
practical remedy available to us.

**Caveat: fragmentation is still only a hypothesis for the ANE penalty.** A
single `select_scatter` on a small tensor is not obviously worth the 2.3 ms gap
(9.5 vs 7.2 ms). The more plausible mechanism is that splitting the delegated
region forces two ANE dispatches plus intermediate transfers, but that has not
been demonstrated, only inferred. Confirming it would need per-op profiling the
benchmark harness does not do.

### Recommendation

- Keep the default `ALL` everywhere; it is correct for every model measured.
- Consider pinning `CPU_ONLY` for ssdlite320 only (24%), after a second device
  confirms it. Single model, single device, and the one fragmented graph.
- Never `CPU_AND_GPU`: fatal on one model, slowest on the other.

### Harness note

A Core ML delegate can abort the process from inside `execute()`. The benchmark
screen now uploads results after EVERY row rather than batching at the end, so a
crash preserves the rows already measured and the last row present identifies
the artifact that died.

---

## 3. Backend-wide toolchain gaps (model-independent)

Reproduced identically on EfficientNet, lfm2.5-embedding and lfm2.5-colbert, so these are
runtime/build gaps, not model problems, and are NOT evidence about quality:

- MLX `8da4w` / `8da8w` -> `Missing out variants: {'torchao::quantize_affine',
  'torchao::dequantize_affine'}`. Activation-quantized paths need out-variant
  kernels the runtime does not register.
- MLX `fpa4w` -> `Could not find libtorchao_ops_mps_aten.dylib`.

Note the asymmetry worth remembering: XNNPACK ships `8da4w` as ColBERT's and the
embedding model's default, but MLX cannot currently export `8da4w` at all.
