"""Export the FaceMesh XNNPACK recipe and exercise its forward contract."""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from executorch.runtime import Runtime

from exporter import registry


def main():
    variant = registry.load("facemesh", backend="xnnpack").default_variant()
    forward = variant.methods["forward"]
    with TemporaryDirectory(prefix="facemesh-smoke-") as out_dir:
        result = subprocess.run(
            [
                sys.executable, "-m", "exporter.cli", "export", "facemesh",
                "--backend", "xnnpack", "--output-dir", out_dir, "--json",
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert json.loads(result.stdout) == {
            "model": "facemesh",
            "results": [{"file": variant.file, "precision": variant.precision}],
        }

        artifact = Path(out_dir) / variant.file
        assert artifact.stat().st_size > 0
        program = Runtime.get().load_program(str(artifact))
        method = program.load_method("forward")
        inputs = [
            torch.zeros(tensor.shape, dtype=getattr(torch, tensor.dtype))
            for tensor in forward.inputs
        ]
        outputs = method.execute(inputs)
        assert len(outputs) == len(forward.outputs)
        for expected, actual in zip(forward.outputs, outputs):
            assert list(actual.shape) == expected.shape, expected.name
            assert actual.dtype == getattr(torch, expected.dtype), expected.name
            assert torch.isfinite(actual).all(), expected.name

    print(f"FaceMesh XNNPACK smoke passed ({variant.file})")


if __name__ == "__main__":
    main()
