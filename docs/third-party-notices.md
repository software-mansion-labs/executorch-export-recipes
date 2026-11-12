# Third-party notices

The root [MIT license](../LICENSE) covers original Software Mansion code and
documentation. It does not replace the licenses of vendored code, dependencies,
model weights, datasets, or generated model artifacts. Existing file headers
and component licenses continue to apply.

## Vendored code with existing notices

| Component | Location in this repository | License notice |
| --- | --- | --- |
| ExecuTorch-derived Meta code | `exporter_models/gemma4/_text_decoder/`, `exporter_models/gemma4_omni/_gemma4/`, `exporter_models/lfm2_5_vl/_lfm2_5_vl/` | [ExecuTorch BSD license](../LICENSE); retain the individual copyright headers. |
| PyTorch neural style transfer | `exporter_models/style_transfer_*/_transformer_net.py` | [PyTorch examples BSD license](../LICENSE). |
| Hugging Face Nemotron feature extraction and cache compression | `exporter_models/nemotron3_diarization/_model.py` | [Apache 2.0](../exporter_models/nemotron3_diarization/THIRD_PARTY_LICENSE); NVIDIA weight terms are separate. |
| Torchvision SSD and SSDLite | `exporter_models/ssdlite320_mobilenet_v3_large/_model/` | Full BSD 3-Clause notices are included in `ssd.py` and `ssdlite.py`. |
| OpenAI Whisper | `exporter_models/whisper/_custom_whisper/` | Full MIT notices are included in the vendored source files. |
| MI-GAN | `exporter_models/migan/_migan.py` | [Upstream MIT license](../LICENSE); retain the attribution in the source. |
| Depth Anything 3 | `exporter_models/depth_anything_v3/_upstream/` | [Apache 2.0 license](../exporter_models/depth_anything_v3/_upstream/LICENSE) and notices in the individual source files. |
| ZipDepth | `exporter_models/zipdepth/_upstream/` | [Upstream MIT license](../exporter_models/zipdepth/_upstream/LICENSE). |

License text sources:

- [ExecuTorch](https://github.com/pytorch/executorch/blob/main/LICENSE)
- [PyTorch examples](https://github.com/pytorch/examples/blob/main/LICENSE)
- [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN/blob/main/LICENSE)

This index records the notices above; it is not an exhaustive inventory of every
dependency or a statement that every model has the same license. Check each
model's upstream source, model card, resolved config's `license` field, and
weight terms before redistributing code or artifacts. Exporting a model does
not grant additional rights to its weights or training data.

When adding vendored code, retain its copyright headers and include its license
and any required notices beside the component. Mark local changes where the
upstream license requires it.
