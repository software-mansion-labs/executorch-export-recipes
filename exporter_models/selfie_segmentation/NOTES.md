# selfie_segmentation notes

- Source: https://github.com/drumichiro/selfie-segmentation-pytorch — a PyTorch
  port of MediaPipe Selfie Segmentation. The model code is vendored in
  `_selfie_segmentation.py`; the `.pth` checkpoints are fetched from the same
  GitHub repo at export time into `weights/` (gitignored), pinned to
  `SOURCE_COMMIT` and checked against `WEIGHTS_SHA256` — the port is an
  unversioned repo, so `master` is not a safe ref to export from.
- The port repo ships no LICENSE file; the license in the config is that of the
  original MediaPipe model (Apache-2.0).
- Output is `(1, 1, H, W)` **sigmoid probabilities** (not logits) — the
  react-native-executorch `BaseSemanticSegmentation` single-channel binary path
  thresholds foreground at 0.5. Input is `[0, 1]` RGB with **no**
  mean/std normalization (pass no `normMean`/`normStd` from JS).
- Two variants, differing only in input size: portrait 256x256 (default) and
  landscape 256x144 (`selfie_segmentation_landscape.pth`). Both share the same
  `precision` (fp32), so `--variant fp32` and the default path always resolve to
  portrait — export the landscape one with `--all-variants`.
- The config is vendored locally even though the HF repo publishes an
  `xnnpack/config.json`: the published one predates the landscape variant and
  lists only portrait. The local config wins over the fetch; keep the two in
  sync when re-publishing.
- No `schema.json` / `get_model_schema` companion: both variants are fully
  static, so the runtime derives the model spec from ExecuTorch `MethodMeta`.
- **The published portrait `.pte` does not hash-match a re-export, and that is
  fine.** `xnnpack/selfie_segmentation_xnnpack_fp32.pte` on the Hub is
  `176aba6a…` / 497 924 B, while the current exporter produces `ea5cd127…` /
  473 732 B (the hash `docs/executorch-1.4.1-bump.md` records for both 1.3.1 and
  1.4.1). Verified 2026-08-21 under the ET 1.4.1 runtime on host: the published
  artifact loads, reports the expected `1x3x256x256 -> 1x1x256x256` contract, and
  produces output **bit-identical** to the re-export (max abs difference exactly
  0.0 over 6 inputs, on both the probability and the logit scale), with both
  matching eager to 2.2e-10. The 24 KB delta is container-level, not numerics, so
  the published build is kept and NOT re-published.
- Note that this model saturates to ~0 on synthetic input (uniform noise, flat
  fields, crude silhouettes all give max probability ~2e-06). Comparing exports
  on random tensors alone is therefore a weak test: agreement on an all-but-zero
  output means little. Compare on the logit scale, or against another `.pte`,
  rather than reading an absolute probability delta.
- If verifying a `.pte` from a numpy image, `np.transpose` leaves the array
  channels-last; the ExecuTorch runtime reads raw buffers as contiguous NCHW —
  `np.ascontiguousarray` first, or the output is garbage.
