# coremltools#2825 - follow-up reproducer

Reply to https://github.com/apple/coremltools/issues/2825#issuecomment-5348173661
("Can you give us a simple self contained code to reproduce the issue?")

## Two corrections to the original report

1. **It is not the CPU cast kernel.** The determinant is whether the model actually
   runs on a **non-CPU** unit. Only the CPU path is safe: `CPU_AND_NE` at fp16 and
   `ALL`/`CPU_AND_GPU` at either precision all segfault, while `CPU_ONLY` is correct
   at both precisions. (`CPU_AND_NE` at fp32 passes only because fp32 cannot run on
   the ANE and falls back to the CPU.) The original stack trace named
   `cast_kernel_cpu`, which pointed the wrong way.
2. **ExecuTorch is not required.** The crash reproduces through plain
   `coremltools.models.MLModel.predict`. The original report said the crash was
   only reachable through ExecuTorch's runtime.

## What actually triggers it

Bisecting the sentence-transformers wrapper, one piece at a time, over a dynamic
sequence dim (`RangeDim(1, 382)`):

| wrapper content | result |
|---|---|
| encoder only (`last_hidden_state`) | OK |
| encoder + CLS/SEP prepend | OK |
| encoder + **masked mean-pool** | **SIGSEGV** |
| encoder + mean-pool + L2 normalize | **SIGSEGV** |

So the trigger is the masked mean-pool, not the encoder:

```python
mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
pooled = torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
```

## Reproducer A - minimal, fully self contained

`minimal.py` below is ~20 lines, no transformers and no ExecuTorch. It does not
segfault, but on the GPU path it reports the underlying complaint, and it is clean
on CPU and with static shapes:

```
Espresso exception: "Invalid blob shape": Data-dependent shapes were disabled: var_39 - [?, 768]
```

Core ML infers the *pooled* output as data-dependent `[?, 768]` even though the
reduction removes the dynamic axis and the true shape is `[1, 768]`. In this small
graph the GPU backend rejects it and falls back; in a full encoder it segfaults.

```python
import numpy as np, torch, coremltools as ct
MAXLEN, HID = 382, 768

class MeanPool(torch.nn.Module):
    def forward(self, tokens, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
        return torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

ex = (torch.randn(1, 8, HID), torch.ones(1, 8, dtype=torch.int32))
ts = torch.jit.trace(MeanPool().eval(), ex)
m = ct.convert(
    ts,
    inputs=[ct.TensorType(name="tokens", shape=ct.Shape(shape=(1, ct.RangeDim(1, MAXLEN), HID)), dtype=np.float32),
            ct.TensorType(name="attention_mask", shape=ct.Shape(shape=(1, ct.RangeDim(1, MAXLEN))), dtype=np.int32)],
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT32,
)
m.save("minimal_dyn.mlpackage")

# CPU_ONLY -> correct. ALL / CPU_AND_GPU -> Espresso "Invalid blob shape".
# (CPU_AND_NE also passes here, but only because this package is fp32 and so cannot
#  reach the ANE -- see the compute-path matrix below.)
mm = ct.models.MLModel("minimal_dyn.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
print(mm.predict({"tokens": np.random.rand(1, 8, HID).astype(np.float32),
                  "attention_mask": np.ones((1, 8), dtype=np.int32)}))
```

## Reproducer B - the actual SIGSEGV

Attached `pool.mlpackage` (416 MB): `all-mpnet-base-v2` + masked mean-pool,
`RangeDim(1, 382)`, `compute_precision=FLOAT32`, iOS18. Five lines to crash it:

```python
import numpy as np, coremltools as ct
m = ct.models.MLModel("pool.mlpackage", compute_units=ct.ComputeUnit.ALL)
n = [i.name for i in m.get_spec().description.input]
m.predict({k: np.ones((1, 8), dtype=np.int32) for k in n})   # SIGSEGV
```

Deterministic, 3/3 runs, at L = 2, 8 and 382. Swap to `ComputeUnit.CPU_ONLY` and the
identical package returns correct output. (`CPU_AND_NE` also passes on this fp32
package, but only via CPU fallback -- at fp16 it segfaults; see the matrix below.)

## Which compute paths are affected

Full matrix for `all-mpnet-base-v2` + masked mean-pool, `RangeDim(1, 382)`, measured
on a **cleared** Core ML compiled-model cache:

| compute_units | precision | result |
| --- | --- | --- |
| `CPU_ONLY` | fp16 | OK |
| `CPU_ONLY` | fp32 | OK |
| `CPU_AND_NE` | fp16 | **SIGSEGV** |
| `CPU_AND_NE` | fp32 | OK - but only because fp32 cannot run on the ANE, so it falls back to the CPU |
| `ALL` | fp16 | **SIGSEGV** |
| `ALL` | fp32 | **SIGSEGV** |

So it is not GPU-specific: **any** non-CPU compute path fails. `CPU_AND_NE` is not a
workaround, it only looks like one when the precision forces a CPU fallback. On the
CPU path the output is exact against the torch reference (cosine 1.000000, max abs
0.000000 at L = 2, 8, 37, 128, 380).

Caveat for anyone reproducing: `~/Library/Caches/executorchcoreml` (or Core ML's own
compiled-model cache) will silently serve an earlier compilation and fake a pass.
Clear it between runs or the matrix above is not reproducible.

`clip-vit-base-patch32`'s text tower, which was also in the original report, runs
**correctly** on `CPU_AND_NE` at both precisions. Whatever the defect is, it is
specific to the mpnet-style graph rather than to every model with a dynamic
sequence dim.

## Environment

coremltools 9.0, macOS 15 arm64 (M4), Python 3.10, torch 2.13.0,
ExecuTorch 1.4.1 (used only to produce reproducer B's package).
