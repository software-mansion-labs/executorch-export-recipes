![ExecuTorch Export Recipes — export, validate, and optimize models for the edge](.github/assets/export-recipes.svg)

# ExecuTorch Export Recipes

**An agent-first harness and collection of model recipes for exporting,
validating, and optimizing AI models for ExecuTorch.**

[![License: MIT](https://img.shields.io/badge/License-MIT-4b6cf4?labelColor=160042)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-7394ff?labelColor=160042)](.python-version)
[![ExecuTorch](https://img.shields.io/badge/Runtime-ExecuTorch-d0e2ff?labelColor=160042)](https://github.com/pytorch/executorch)

[Quick start](#quick-start) · [Models](#model-collection) · [Agent workflow](#working-with-an-agent) · [CLI](#cli) · [Layout](#layout) · [Contract](#the-exportpy-contract)

Bring a checkpoint, pick a backend, and build an ExecuTorch `.pte` using a
model-owned export script and pinned environment. The CLI discovers recipes,
resolves their I/O specifications, and runs exports. The skills guide agents
through failures, output validation, profiling, and optimization.

Built by [Software Mansion](https://swmansion.com/) for the model ecosystem
around [React Native ExecuTorch](https://github.com/software-mansion/react-native-executorch).
The exported artifacts use ExecuTorch and can also serve other compatible
ExecuTorch integrations.

## What lives here

| Piece | Purpose |
| --- | --- |
| [Export harness](src/exporter/) | Discover models, select sizes/backends/variants, resolve configs, and invoke the owning export script. |
| [Model collection](exporter_models/) | Model code, wrappers, dependency locks, backend recipes, and notes from real exports. |
| [Agent skills](skills/) | Procedures for adding recipes, recovering failed exports, checking outputs, and measuring speed. |

**Writing a `.pte` is the start of verification.** The CLI does not automatically
run numerical parity checks or target-device benchmarks. A recipe's presence
does not mean every backend/variant is verified on every device; read the
model's `NOTES.md` and the relevant backend skill.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and
[Git LFS](https://git-lfs.com/). Use Python **3.10** for the harness and model
environments. Current model locks target **macOS on Apple Silicon**; other
hosts may require separate dependency resolution and validation.

```bash
git lfs install
git clone https://github.com/software-mansion-labs/executorch-export-recipes.git
cd executorch-export-recipes

# Discover the available models without installing torch or a backend.
uv run --locked exporter list --json

# Inspect a recipe and its export plan.
uv run --locked exporter describe facemesh --backend xnnpack
uv run --locked exporter export facemesh --backend xnnpack --dry-run

# Install this model's pinned stack and export its default variant.
uv run --locked --project exporter_models/facemesh \
  exporter export facemesh --backend xnnpack \
  --output-dir exports/facemesh/xnnpack
```

Some recipes download gated checkpoints and require Hugging Face access.
Vulkan and some MLX recipes require a matching source-built ExecuTorch fork;
their dependency pins document the stack but are not a working `uv` build
recipe. Follow the [source-build instructions](skills/backends/vulkan/SKILL.md#environment-the-fork-pin-is-not-uv-buildable)
and the model's notes for those exports.

## Model collection

Browse [all model directories](exporter_models/) or run `exporter list` for the
current catalog. Examples include:

| Workload | Example recipes |
| --- | --- |
| Text generation | [Gemma 4](exporter_models/gemma4/), [LFM2.5](exporter_models/lfm2_5_text/), [Qwen 3](exporter_models/qwen3/), [Llama 3.2](exporter_models/llama3_2/) |
| Vision and multimodal | [LFM2.5 VL](exporter_models/lfm2_5_vl/), [Gemma 4 Omni](exporter_models/gemma4_omni/), [CLIP](exporter_models/clip_vit_base_patch32_vision/) |
| Detection and segmentation | [YOLO26](exporter_models/yolo26/), [RF-DETR](exporter_models/rfdetr_nano_detector/), [FastSAM](exporter_models/fast_sam/) |
| Speech | [Whisper](exporter_models/whisper/), [Moonshine](exporter_models/moonshine/), [Kokoro](exporter_models/kokoro/), [FSMN VAD](exporter_models/fsmn_vad/) |
| OCR and document processing | [PaddleOCR](exporter_models/paddleocr/), [EasyOCR](exporter_models/easyocr/), [document layout](exporter_models/paddleocr_layout/) |
| Embeddings and image processing | [LFM2.5 embeddings](exporter_models/lfm2_5_embedding_350m/), [Depth Anything V3](exporter_models/depth_anything_v3/), [MI-GAN](exporter_models/migan/), [style transfer](exporter_models/style_transfer_mosaic/) |

Backend availability is declared per model in `model.json`:

| Backend | Intended target | Recipe guidance |
| --- | --- | --- |
| XNNPACK | CPU on Android and Apple platforms | [XNNPACK skill](skills/backends/xnnpack/SKILL.md) |
| CoreML | Apple platforms; compute-unit behavior needs device checks | [CoreML skill](skills/backends/coreml/SKILL.md) |
| MLX | Apple GPU; the iOS delegate requires a physical device | [MLX skill](skills/backends/mlx/SKILL.md) |
| Vulkan | Android GPU; verify against the shipping runtime and GPU limits | [Vulkan skill](skills/backends/vulkan/SKILL.md) |

Published artifacts are linked from each model's `model.json` `hf` field and
collected on [Software Mansion's Hugging Face organization](https://huggingface.co/software-mansion).
New generated weights and exports belong outside Git. Git LFS is needed for
large files retained in the migrated history.

## Working with an agent

Point your coding agent at [AGENTS.md](AGENTS.md) and the relevant skill. These
are repository-local instructions; no particular agent runtime is required.

```text
Read AGENTS.md and skills/adding-a-new-export/SKILL.md.
Add an XNNPACK recipe for <model>, using the model's pinned environment.
If export fails, follow skills/recover-export/SKILL.md.
Validate outputs and benchmark candidates using
skills/validate-and-optimize/SKILL.md. Keep the baseline, document thresholds,
and report which host/device checks actually passed.
```

| Task | Start here |
| --- | --- |
| Add a model, size, or backend | [Adding an export](skills/adding-a-new-export/SKILL.md) |
| Understand configs, layouts, and dependencies | [Model conventions](skills/model-conventions/SKILL.md) |
| Recover an export or runtime failure | [Export recovery](skills/recover-export/SKILL.md) |
| Verify outputs and improve inference speed | [Validation and optimization](skills/validate-and-optimize/SKILL.md) |

The workflow is **baseline → export → validate → profile → optimize → revalidate**.
Measure candidates on the same target device and inputs. Keep the accuracy
budget fixed, and reject faster candidates that fail it.

## Layout

```text
exporter_models/<model>/
  model.json                  # source/published repos, config path, sizes, backends
  pyproject.toml + uv.lock     # this model's verified dependency stack
  <backend>/export.py          # export(config, out_dir) writes the artifact(s)
  <backend>/config.json        # optional local override of a fetched spec
  <size>/<backend>/config.json # optional sized config, per model.json template
  _vendored_code.py            # model-owned source and export adaptations
  NOTES.md                    # quirks, validation evidence, benchmarks
```

Each backend owns one export script across sizes. `model.json` selects the
config using `{size}` and `{backend}` placeholders; a local config at the
resolved path takes precedence over Hugging Face. Each model owns its code
and dependencies so incompatible stacks can coexist.

## The export.py contract

```python
def export(config: dict, out_dir: str) -> None:
    """Export and lower the model; write the configured .pte file(s)."""
```

The return value is advisory. The harness reports the output names declared
by the spec. Keep top-level imports to stdlib and torch; import backend
libraries and path-load vendored code inside `export()`.

## CLI

```text
exporter list [--json]
exporter describe <model> [--size S] [--backend B] [--json]
exporter export <model> [--size S] [--backend B]
                [--variant PRECISION | --all-variants]
                [--local-spec FILE] [--output-dir DIR] [--dry-run] [--json]
```

`--variant` selects by precision; omitted size/backend/variant selections use
the model's defaults. Multi-PTE models emit their full variant set together.
`describe` and `--dry-run` resolve the spec and plan without importing the
export script, but may fetch a remote config. To run without the installed
CLI, use `PYTHONPATH=src python -m exporter.cli` in the intended environment.

## Contributing

Start with the [model conventions](skills/model-conventions/SKILL.md). Pin a
reproducible stack, preserve upstream licenses, and keep each recipe self-contained.
Include the actual export command and the validation/benchmark evidence in the
model's notes. Document unavailable devices or evaluation data explicitly.

Validate exported models against reference outputs and test them on the target
runtime using the [validation skill](skills/validate-and-optimize/SKILL.md).
Successfully producing a `.pte` file does not establish correctness.

## License

Original Software Mansion code and documentation are licensed under [MIT](LICENSE).
Vendored code retains its upstream licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
Model weights, datasets, and exported artifacts remain subject to their own
terms. This repository's license does not relicense those materials.
