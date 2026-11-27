# Echo Android demo

Records a reference voice, then speaks typed text in it. Native engine in `cpp/` (sampler, codec,
KV handling), a single-activity UI in `java/`; `cpp/echo_cli.cpp` is the same engine as an adb CLI.

## Models

Put these in the app's files dir, `/sdcard/Android/data/com.swmansion.echodemo/files/`
(`adb push`). Run the commands from `exporter_models/echo`; QNN ones need the QNN environment in
`../NOTES.md`, the rest this dir's uv project (`uv run python ...`).

| file | from |
| --- | --- |
| `tokenizer.json` | source repo (software-mansion/Echo), root |
| `codec_decoder_xnnpack_fp32.pte` | `exporter export echo --backend xnnpack` (`../NOTES.md`, Reproduce) |
| `codec_encoder_xnnpack_fp32_multi.pte` | `python android/export_encoder.py <file>` |
| `echo_vulkan_8da4w.pte`, `echo_vulkan_fp16.pte` | `exporter export echo --backend vulkan` |
| `echo_qnn_fp16_t256.pte` | `python qnn/export_qnn.py <file> --precision fp16` |
| `echo_qnn_a16w8_p320_t256.pte` (optional) | `qnn/calibrate.py`, then `qnn/export_qnn.py <file> --precision a16w8 --calib ...` |
| `echo_codec_qnn_a16w8_t256.pte` | `python qnn/export_codec_qnn.py <file> --precision a16w8` |

The app lists the backbones whose files are present (`MainActivity.BACKBONES`). The NPU options use
the QNN backbone with the smallest prefix bucket that fits the prompt and the static QNN codec for
clips up to 10 s (256 frames); longer texts fall back to the Vulkan backbone and the XNNPACK decoder.

## Build

```
RNE_LIBS=<dir> QAIRT_ROOT=<qairt/2.47.0.260601> FBJNI_SO=<libfbjni.so> ./build.sh
adb install -r build/echo-demo-release.apk
```

`RNE_LIBS` holds react-native-executorch's v0.11.1-libs release extracted (`gh release download
v0.11.1-libs -R software-mansion/react-native-executorch -p '*-android-arm64-v8a.tar.gz' -p
headers.tar.gz`, then untar each). Details in the `build.sh` header.

## CLI

Same engine without the UI, for timing on a phone over adb: build `cpp/echo_cli.cpp` +
`cpp/echo_engine.cpp` with the NDK against the same libs, push the libs next to it (the core lib
also as `libexecutorch_jni.so`, its SONAME), and run from a dir holding the models in `m/`:

```
export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH="$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp"
ECHO_DECODER=echo_codec_qnn_a16w8_t256.pte ECHO_DECODER_FRAMES=256 ./echo_cli m echo_qnn_fp16_t256.pte \
  ref_24k_f32.raw ref.txt text.txt en 250 7 out_f32.raw echo_vulkan_8da4w.pte
```

It prints prefix / sampling / codec times. Galaxy S26 Ultra, 184-token prompt, 250 frames (10 s):
26-34 + 270-290 + 45 ms.
