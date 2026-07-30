# gemma4 — ExecuTorch export for Gemma 4 E2B

Self-contained model definition and export recipes for Gemma 4 E2B on
ExecuTorch, targeting XNNPACK (CPU) and Vulkan (GPU), text-only and multimodal
(text + vision + audio).

Four recipes, each reproducing a PTE that was actually built:

| Recipe | Backend | Modalities | On-device validated |
|---|---|---|---|
| `xnnpack-text` | XNNPACK | text | yes |
| `xnnpack-mm` | XNNPACK | text + vision + audio | yes |
| `vulkan-text` | Vulkan | text | yes |
| `vulkan-mm` | Vulkan (audio on XNNPACK) | text + vision + audio | **no** |

Read `docs/RECIPES.md` before trusting `vulkan-mm` — it carries a one-constant
fix for a known failure in the currently-shipping PTE, and that fix has never
been run on a device.

The validator refuses the shipping PTE's own configuration, on purpose:

```
$ python -m gemma4.export --recipe vulkan-mm --text-decoder-prefill-len 128 --dry-run
  ERROR [R5-seq-bound] text_decoder seq bound (128) is below the soft-token
  count the encoders emit (vision 280, audio 744; need >= 744) ...
refusing to export: 1 validation error(s).
```

---

## Quickstart

```bash
conda activate gemma4
cd /path/to/executorch

# See what a recipe would do, without exporting.
PYTHONPATH=. python -m gemma4.export --recipe vulkan-mm --dry-run

# Export.
PYTHONPATH=. python -m gemma4.export \
    --recipe xnnpack-mm \
    --checkpoint gemma_export/gemma4_e2b \
    --out build/gemma4_xnnpack_mm.pte
```

`--dry-run` resolves the recipe, applies your overrides, runs every validation
rule and prints the result. It builds nothing and takes a second. Use it first.

### The four commands

```bash
PYTHONPATH=. python -m gemma4.export --recipe xnnpack-text --out build/g4_xnnpack_text.pte
PYTHONPATH=. python -m gemma4.export --recipe xnnpack-mm   --out build/g4_xnnpack_mm.pte
PYTHONPATH=. python -m gemma4.export --recipe vulkan-text  --out build/g4_vulkan_text.pte
PYTHONPATH=. python -m gemma4.export --recipe vulkan-mm    --out build/g4_vulkan_mm.pte
```

### Common overrides

```bash
--checkpoint PATH               # or set GEMMA4_CHECKPOINT
--num-layers N                  # truncate for a smoke test; validate at 5 before 35
--max-seq-len N                 # KV cache depth
--prefill-len N                 # token_embedding / forward seq bound
--text-decoder-prefill-len N    # text_decoder seq bound — this sizes the arena
--group-size N                  # linear quant group size
--ple-bits {2,4}                # PLE embedding bit width
--force                         # export despite validation errors (rarely right)
--list                          # print all recipes and exit
```

---

## Prerequisites

| Item | Value |
|---|---|
| conda env | `gemma4` |
| Python | 3.12.13 |
| torch | 2.11.0 |
| torchao | 0.15.0+git9338966da |
| transformers | 5.7.0 (must provide `models/gemma4`) |
| executorch | installed wheel, built from a tree with the Vulkan kernel patches |
| checkpoint | `gemma_export/gemma4_e2b/model.safetensors` (10,246,621,918 B) |

Base conda `transformers` has no `gemma4` at all — it must be the `gemma4` env's
copy. There is also a vendored clone at `<repo>/transformers` whose
`modeling_gemma4.py` differs by md5; it is **not** what loads (verified: it does
not shadow site-packages under `PYTHONPATH=.`) and its line numbers should not
be consulted.

### Import order is load-bearing

```python
from executorch.runtime import Runtime          # FIRST
import executorch.kernels.quantized             # SECOND
import executorch.extension.llm.custom_ops.custom_ops
```

Reversed, `embedding_{2,4}bit.out` silently fails to register and `load_method`
fails later with an unhelpful error. `gemma4/export/__main__.py` enforces this.

### Vulkan backend kernels

The Vulkan recipes depend on patched `backends/vulkan/` kernels — the softmax
max-subtraction fix, the gelu/tanh fp16 clamp, `GemmaSDPA.cpp` implementing
`et_vk::gemma_sdpa_with_kv_cache` and `et_vk::gemma_custom_sdpa`, and several
`resize_fn` fixes for dynamic shapes. These live in a **separate repository**
and are not vendored here.

Export and runtime must come from the same tree state. A PTE built against one
set of shaders and run against another produces gibberish at runtime, not an
error — this has cost multiple debugging days and is the first thing to check
when on-device output is nonsense.

---

## Layout

```
gemma4/
├── model/                model definition — no ExecuTorch dependency except
│   ├── text.py           where noted
│   ├── vision.py
│   ├── audio.py
│   ├── custom_sdpa_ops.py    registers the two et_vk:: ops (import for side effect)
│   ├── mel_frontend.py
│   ├── truncated.py      N-layer truncation rig for smoke tests
│   └── transforms/       source transforms applied before export
├── export/
│   ├── config.py         typed Recipe / QuantSpec / LengthSpec / AttnFlags
│   ├── presets.py        the four recipes
│   ├── validate.py       pre-flight rules
│   ├── patches.py        gated monkey-patches
│   ├── quant.py          quantisation + the Vulkan fp16 retarget
│   ├── wrappers.py       per-method export nn.Modules
│   ├── pipeline.py       build → transform → quantize → export → lower → write
│   └── __main__.py       CLI
├── assets/e2b/           model config JSON + chat template (72 KB)
├── tests/                port parity test
└── docs/
    ├── RECIPES.md        per-recipe detail, provenance, sequence-length semantics
    ├── LEVERS.md         confirmed / falsified / unclaimed optimisations
    └── MODEL_INTERFACE.md  the PTE's runtime contract for runner authors
```

`assets/e2b/` deliberately excludes `tokenizer.json` (32 MB) — it lives in the
checkpoint directory, which you need anyway, and that is what gets pushed to the
device.

---

## Workflow notes

**Truncate before you commit to a full export.** `--num-layers 5` builds in a
fraction of the time and catches most graph-level breakage. Validate the
truncated PTE against eager before running the 35-layer export.

**Validate in eager Python before exporting.** Most hypotheses about numerics
can be tested without an export at all, and an export cycle is expensive
(multi-GB, RAM-bound, one at a time — two concurrent exports will not fit).

**Do not retry on a Vulkan VRAM OOM.** Arena OOM at `load_method` is a sizing
constraint, not a transient failure. Retrying wastes hours. Lower a prefill
bound, or shrink the graph.

**Arena bytes, not partition count, drive `load_method` OOM.** A build with 37
partitions and a 91 MB arena OOM'd where one with 36 partitions and a 66 MB
arena passed.

---

## Relationship to `gemma_export/`

This package is a **copy**. `gemma_export/` is untouched and remains the
archive: ~166 GB of PTEs, tensor dumps, logs and roughly 460 experiment scripts,
including all the negative results.

The port is verified bit-identical to the archive:

```bash
PYTHONPATH=. python gemma4/tests/test_port_parity.py
```

Three deliberate differences:

1. **Lever G removed.** `use_rotate_half_matmul` and `make_rotate_half_basis`
   are gone — a measured regression (+11 host ops at 5L, projected +77 at 35L).
   See `docs/LEVERS.md`.
2. **`kv_quant.py` no longer sideloads.** The archive read Llama's
   `custom_kv_cache.py` off disk via `importlib` with four fabricated
   `sys.modules` stubs, at a path derived from `__file__`. The installed wheel
   exports that module normally, so it is now a plain import.
3. **Real package.** The archive had no `__init__.py` anywhere and resolved only
   as a namespace package because the repo root happened to be on `sys.path`.
