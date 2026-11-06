# Echo Android demo

Records a reference voice, then speaks typed text in it. Native engine in `cpp/` (sampler, codec, KV
handling), a single-activity UI in `java/`; `cpp/echo_cli.cpp` is the same engine as an adb CLI.

Build: `ET_DIR=... QNN_LIBS=... ./build.sh` (see the script header). Models go in the app's external files dir
(`/sdcard/Android/data/com.swmansion.echodemo/files/`): `tokenizer.json`, the backbones listed in
`MainActivity.BACKBONES`, `codec_dec_xnnpack.pte`, `codec_enc_xnnpack_multi.pte`, and for the NPU options
`echo_codec_qnn_a16w8_t256.pte` (`../qnn/`).

The NPU options run the QNN backbone (`../qnn/export_qnn.py`), with the smallest prefix bucket that fits the
prompt, and the static QNN codec for clips up to 10 s; longer texts fall back to the Vulkan backbone and the
XNNPACK decoder.
