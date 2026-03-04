# SSDLite export

This directory contains the export scripts for the
[SSDLite](https://docs.pytorch.org/vision/main/models/ssdlite.html) object
detection model.

NOTE: The original implementation in PyTorch had to be modified to make it
possible to export the model. In particular, the non-maximum suppression (NMS)
postprocessing step has been reimplemented. For details see:
[`ssd.py`](model/ssd.py).

NOTE: I encountered severe difficulties when trying to quantize the model to
`int8` for xnnpack backend which I have not been able to overcome. As of right
now the model is only available in `fp32` precision for xnnpack and `fp16` or
`fp32` for coreml backend.

## How to run

For reproducible results this repository uses [`uv`](https://docs.astral.sh/uv/)
to manage dependencies. To install dependencies run:

```bash
uv sync
```

The Jupyter notebooks contain the code to export the models to Executorch
format.
