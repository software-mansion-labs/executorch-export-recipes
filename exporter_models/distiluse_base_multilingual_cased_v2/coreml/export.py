import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import Dim, export as torch_export

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
_MAX_TOKENS = 126  # model max_seq_length 128 - 2 for CLS/SEP


def _load_model():
    name = "distiluse_coreml_model"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_distiluse_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]

    module = _load_model().load_embedder()
    inputs = (
        torch.zeros(1, 8, dtype=torch.int64),
        torch.ones(1, 8, dtype=torch.int64),
    )
    dim = Dim("tokens", min=1, max=_MAX_TOKENS)
    dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}

    ep = torch_export(module, inputs, dynamic_shapes=dynamic)

    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")

    # CPU_ONLY, deliberately. These graphs fail the Apple Neural Engine path
    # (coremltools#2825 masked mean-pool over a dynamic sequence dim, surfacing
    # as `Espresso exception: "Invalid blob shape": Data-dependent shapes were
    # disabled`; clip-text additionally `ANECCompile() FAILED`). Core ML still
    # returns correct output because it silently falls back to CPU, but the
    # failed ANE compile is paid on every call. Pinning CPU_ONLY skips the
    # attempt entirely. Measured on an iPhone 16, fp16, 3 warmup + 15 timed:
    #   all-MiniLM      13.7 ms -> 2.5 ms   5.5x
    #   multi-qa        14.1 ms -> 2.5 ms   5.6x
    #   paraphrase-L12  26.9 ms -> 4.2 ms   6.4x
    #   distiluse       33.0 ms -> 8.0 ms   4.1x
    #   clip-text       18.1 ms -> 4.9 ms   3.7x
    specs = CoreMLBackend.generate_compile_specs(
        compute_unit=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )
    program = to_edge_transform_and_lower(
        {"forward": ep},
        partitioner=[CoreMLPartitioner(compile_specs=specs)],
        constant_methods={"get_model_schema": schema_json},
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
