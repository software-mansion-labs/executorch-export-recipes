# Semantic Segmentation export

This repository contains scripts to export semantic segmentation models listed
[here](https://docs.pytorch.org/vision/main/models.html#semantic-segmentation)
to Executorch format:

- [fcn_resnet50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html#fcn-resnet50)
- [fcn_resnet101](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet101.html#fcn-resnet101)
- [deeplabv3_resnet50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html#deeplabv3-resnet50)
- [deeplabv3_resnet101](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#deeplabv3-resnet101)
- [deeplabv3_mobilenet_v3_large](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html#deeplabv3-mobilenet-v3-large)
- [lraspp_mobilenet_v3_large](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html#lraspp-mobilenet-v3-large)

## How to run

For reproducible results this repository uses [`uv`](https://docs.astral.sh/uv/)
to manage dependencies. To install dependencies run:

```bash
uv sync
```

The Jupyter notebook contains the code to export the models to Executorch
format.