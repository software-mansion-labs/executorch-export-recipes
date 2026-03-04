# Style Transfer exports

This directory contains the export scripts for the style transfer models with
following styles:

- candy,
- mosaic,
- rain_princess,
- udnie.

The original code and model definition is from the PyTorch examples repository:
<https://github.com/pytorch/examples/tree/main/fast_neural_style>

## How to run

For reproducible results this repository uses [`uv`](https://docs.astral.sh/uv/)
to manage dependencies. To install dependencies run:

```bash
uv sync
```

The Jupyter notebooks contain the code to export the models to Executorch
format.
