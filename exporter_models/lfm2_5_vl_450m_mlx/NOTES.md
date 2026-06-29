# lfm2_5_vl_450m_mlx notes

MLX variant of LFM2.5-VL-450M. See `../lfm2_5_vl_450m/NOTES.md` for the full
rationale. Summary:

- **Local config.json** (not a model.json pointer): the 450M size is unpublished
  on HF — only `react-native-executorch-lfm2.5-VL-1.6B` exists. Hand-authored
  multi-method VL contract until a 450M repo is published.
- Model code vendored under `_lfm2_5_vl/` (VL recipe; shared llama infra kept as
  import). `export.py` passes the resolved 450M params_path (dim 1024).
- Structural-only: the MLX backend needs an executorch source build, so the real
  export runs there, not in the harness's verification env.
