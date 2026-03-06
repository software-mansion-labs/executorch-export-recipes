# Style Transfer exports

This directory contains the export scripts for the style transfer models with
following styles:

- candy,
- mosaic,
- rain_princess,
- udnie.

The original code and model definition is from the PyTorch examples repository:
<https://github.com/pytorch/examples/tree/main/fast_neural_style>

`transformer_net.py` is modified from the original PyTorch implementation for
better ExecuTorch/XNNPACK inference performance:

- Reflection padding (`_reflect_pad`). `torch.nn.ReflectionPad2d` is replaced
  with a custom `_reflect_pad` function that implements the same operation using
  `flip` + `cat`. The `flip` + `cat` formulation exposes the operation as
  primitives that are handled more efficiently by the delegate.

## How to run

For reproducible results this repository uses [`uv`](https://docs.astral.sh/uv/)
to manage dependencies. To install dependencies run:

```bash
uv sync
```

The Jupyter notebooks contain the code to export the models to Executorch
format.
