"""Diagnostic PT2E interpreter matching XNNPACK's per-token dynamic scales.

XNNPACK Linear uses num_nonbatch_dims=1. The default PT2E dynamic
QuantizationSpec is per_tensor_affine, so its eager chooser has different
activation granularity for rank-3 sequence tensors. Keep the converted weights
and every model operation; evaluate only activation Q/DQ operations row by row.
This is a reference evaluator, never an export/model/runtime patch.
"""
import torch


class RowwiseDynamicReference(torch.fx.Interpreter):
    def call_function(self, target, args, kwargs):
        name = str(target)
        if name == 'quantized_decomposed.choose_qparams.tensor':
            x, *rest = args
            pairs = [target(row, *rest, **kwargs) for row in x.reshape(-1, x.shape[-1])]
            shape = (*x.shape[:-1], 1)
            return tuple(torch.stack([p[i].reshape(()) for p in pairs]).reshape(shape) for i in range(2))
        if name in ('quantized_decomposed.quantize_per_tensor.tensor',
                    'quantized_decomposed.dequantize_per_tensor.tensor'):
            x, scale, zero, *rest = args
            if scale.numel() > 1:
                rows = x.reshape(-1, x.shape[-1])
                scales, zeros = scale.reshape(-1), zero.reshape(-1)
                return torch.stack([target(row, scales[i], zeros[i], *rest, **kwargs)
                                    for i, row in enumerate(rows)]).reshape(x.shape)
        return super().call_function(target, args, kwargs)


def main():
    import argparse
    from pathlib import Path
    # Register decomposed quantization operators before deserializing the EP.
    import torchao.quantization.pt2e.quantize_pt2e  # noqa: F401
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', required=True)
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with torch.inference_mode():
        program = torch.export.load(args.program)
        interpreter = RowwiseDynamicReference(program.module())
        fixtures = torch.load(args.fixtures, weights_only=True)
        result = []
        for inputs, _ in fixtures:
            outputs = interpreter.run(*inputs)
            result.append((inputs, tuple(x.clone() for x in outputs)))
        torch.save(result, Path(args.output))


if __name__ == '__main__':
    main()
