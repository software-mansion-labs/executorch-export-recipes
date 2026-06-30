# gemma4_mlx notes

Gemma 4 E2B text decoder, MLX backend, int4. Model code vendored under
`_text_decoder/` (bit-for-bit copy of the executorch
`examples/models/gemma4/text_decoder/` package); `export.py` transcribes the body
of `export_gemma4_e2b_mlx`.

## Export settings are the on-device-tested invocation (not function defaults)

The recipe is the executorch `export_mlx_standalone.py`, whose `export()` calls
`export_gemma4_e2b_mlx` with the BAKED-IN, on-device-tested settings (its docstring:
"int4 / group_size 64, bf16, compile-friendly"):

- `dtype="bf16"`, `qlinear="4w"`, `qembedding="4w"`, `group_size=64`,
  `compile_friendly=True`, `max_seq_len=2048`.

These are NOT the `export_gemma4_e2b_mlx` function-signature defaults (fp32 / gs32 /
compile_friendly=False) — they are the values the standalone passes, which produced
the published `gemma4_e2b_mlx_int4.pte`. `export.py` bakes these same values, so it is
faithful to the recipe's actual invocation.

Structural-only: the MLX backend needs an executorch source build, so the real export
runs there (the published .pte already exists on HF).
