# mobilenet_v2 notes

- Purpose: this model exists as the **proof-of-concept for the QNN backend**
  (react-native-executorch#1333), not as a model we ship. MobileNetV2 was picked
  because it is small, quantizes cleanly to 8a8w on the HTP, and upstream
  ExecuTorch ships a reference runner for it, so a produced `.pte` can be
  checked against known top-1/top-5 instead of being trusted blind.
- No `hf` / `repo` in `model.json`: nothing is published for it, so the config
  is local and the weights come from torchvision
  (`MobileNet_V2_Weights.IMAGENET1K_V2`).
- **The export does not run on macOS.** The QNN AOT compiler ships host
  libraries for `x86_64-linux-clang` and `x86_64-windows-msvc` only; there is no
  macOS host library. Export on a Linux x86_64 box (or WSL); only the phone is
  shared. Full setup in `skills/backends/qnn`.
- **A QNN `.pte` is compiled for one specific SoC.** The chipset comes from
  `QNN_SOC_MODEL` (default `SM8650`, Snapdragon 8 Gen 3) because it is not part
  of the model spec, and the variant filename does not carry it — export each
  target into its own directory (`--output-dir out/SM8650`) or the next SoC
  overwrites the previous artifact.
- int8 calibration reads `QNN_CALIBRATION_DIR` (a folder of images, e.g. the
  imagenet-mini val split) and `QNN_CALIBRATION_SIZE` (default 100). With no
  directory it calibrates on random noise and warns: the `.pte` still exports and
  runs on the HTP, which is enough to prove the toolchain, but its predictions
  are worthless.
- The fp16 variant is a compile-time HTP precision, not a quantization pass, and
  **HTP fp16 needs a premium-tier SoC** (Snapdragon 8 Gen 1 or newer). int8 is
  the variant to trust on older chips.
- Input is `[0, 1]` RGB `[1,3,224,224]`; the ImageNet mean/std normalization is
  baked into the graph, matching `efficientnet_v2_s` and the
  react-native-executorch classification contract (pass no `normMean`/`normStd`
  from JS). Calibration images are therefore fed unnormalized too.
- No `schema.json` / `get_model_schema` companion: the graph is fully static, so
  the runtime derives the model spec from ExecuTorch `MethodMeta`.
- Running the resulting `.pte` **inside the react-native-executorch app is a
  separate task** — RNE's native build is XNNPACK/CoreML only and would need the
  QNN delegate plus the QNN runtime `.so`s and Hexagon skels in the APK. The
  milestone here is a validated on-device `.pte` (see the skill's validation
  section).
