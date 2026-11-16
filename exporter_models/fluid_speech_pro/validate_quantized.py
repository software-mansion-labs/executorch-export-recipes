"""Compare loaded quantized encoder with its saved converted-graph fixtures.

Only load reference files generated locally by the trusted export recipe.
Original-source task quality must also be measured using benchmark.py.
"""
import argparse
import json
import sys
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pte', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    from executorch.runtime import Runtime
    from executorch.extension.pybindings.portable_lib import _unsafe_reset_threadpool
    torch.set_num_threads(4)
    _unsafe_reset_threadpool(4)
    program = Runtime.get().load_program(args.pte)
    method = program.load_method('forward')
    report = dict(metadata=str(method.metadata), checks=[], passed=False)
    try:
        with torch.inference_mode():
            for inputs, expected in torch.load(args.reference, weights_only=True):
                actual = method.execute([x.contiguous() for x in inputs])
                assert len(actual) == len(expected)
                for slot, (got, ref) in enumerate(zip(actual, expected)):
                    assert got.shape == ref.shape and got.dtype == ref.dtype
                    assert torch.isfinite(got).all() and torch.isfinite(ref).all()
                    if not got.is_floating_point():
                        assert torch.equal(got, ref)
                        report['checks'].append(dict(frames=inputs[0].shape[1], slot=slot, exact=True))
                        continue
                    diff = (got-ref).abs()
                    bad = int((~torch.isclose(got, ref, atol=1e-3, rtol=1e-3)).sum())
                    row = dict(frames=inputs[0].shape[1], slot=slot, max_abs=float(diff.max()),
                               max_relative=float((diff/ref.abs().clamp_min(1e-6)).max()),
                               outside_budget=bad, atol=1e-3, rtol=1e-3)
                    report['checks'].append(row)
                    print(row, flush=True)
                    # Retain every boundary result, including failures, for diagnosis.
        report['passed'] = all(x.get('outside_budget', 0) == 0 for x in report['checks'])
    finally:
        Path(args.report).write_text(json.dumps(report, indent=2))
    if not report['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
