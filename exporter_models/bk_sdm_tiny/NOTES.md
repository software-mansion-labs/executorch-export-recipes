# bk_sdm_tiny notes

BK-SDM-tiny diffusion (text_encoder + UNet + VAE), 5 .pte variants via the multi-pte
path. `_export_utils.py` vendors the recipe's model-export helpers
(`exports/diffusers/export_utils.py`); `export.py` transcribes
`exports/diffusers/export_pipeline.py`, loading from `vivym/bk-sdm-tiny-vpred`.

## Intentional scope vs the recipe

- **Both image sizes (512 and 256)** are exported for UNet and VAE — the recipe's
  `export_pipeline` takes one `--image_size`; the harness produces both because the
  PUBLISHED config has all 5 variants (`unet` + `unet_256`, `vae` + `vae_256`,
  text_encoder). All 5 verified `io_ok` + `parity_ok`.
- **Tokenizer + scheduler export dropped.** The recipe also calls `export_tokenizer`
  and `export_scheduler` — those produce non-`.pte` assets (tokenizer.json /
  scheduler config), not model `.pte`s, and are already shipped in the HF repo. The
  harness exports only the 5 model `.pte`s it verifies against the config.

## Core ML / MLX: neither backend, by decision

This model ships **XNNPACK only** -- no Core ML and no MLX export exists, and neither
has been attempted.

This is a prioritisation decision, not a technical finding: during the 2026-08 Apple
backend sweep this model was explicitly deprioritised in favour of the higher-traffic
computer-vision and embedding models. Nothing is known to block a Core ML or MLX
export here. If you want one, start from a sibling model's `coreml/export.py` --
there is no recorded failure to work around.
