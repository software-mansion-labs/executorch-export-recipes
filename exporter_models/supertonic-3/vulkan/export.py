"""Supertonic 3 -> ExecuTorch Vulkan (multi_pte, fp16).

Same four sub-models and the same dynamic bounds as the XNNPACK export; only
the partitioner and the precision differ, so the model plumbing is imported
from ``xnnpack/export.py`` rather than duplicated.

On a Galaxy S26 Ultra (Adreno 840) this is 1.5-2.1x faster than the XNNPACK
build per sub-model, and 2.1x end to end over the eight flow-matching steps.
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch

from exporter.schema import load_and_validate_schema

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _xnnpack_module():
    """The XNNPACK exporter, loaded as a module for its model plumbing.

    _load_convert / _download_onnx_dir / _example_inputs / _method_spec /
    _size_oblivious_export / _submodel_for are backend independent.
    """
    name = "supertonic3_vulkan_via_xnnpack"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _MODEL_DIR / "xnnpack" / "export.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str):
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    xn = _xnnpack_module()

    convert = xn._load_convert()
    models = convert.load_models(onnx_dir=xn._download_onnx_dir())
    all_schemas = xn._with_t_bound(json.loads(load_and_validate_schema(_MODEL_DIR / "schema.json")))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Size-oblivious mode has to wrap both export and lowering; see the comment
    # on _size_oblivious_export in the XNNPACK exporter.
    with xn._size_oblivious_export(), torch.inference_mode():
        ex = xn._example_inputs()
        specs = xn._method_spec(models, ex)

        for variant in config["variants"]:
            name = xn._submodel_for(variant["file"])
            model, inputs, dynamic_shapes = specs[name]

            ep = torch.export.export(
                model, inputs, dynamic_shapes=dynamic_shapes, strict=False
            )

            program = to_edge_transform_and_lower(
                {"forward": ep},
                # fp16 throughout. Every sub-model stays within 6e-4 relative of
                # its fp32 CPU reference (cosine 0.9994 or better on device).
                partitioner=[VulkanPartitioner(compile_options={"force_fp16": True})],
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
                ),
                constant_methods={
                    "get_model_schema": json.dumps({"forward": all_schemas[name]})
                },
            ).to_executorch()

            out_path = out_dir / variant["file"]
            out_path.write_bytes(program.buffer)
            written.append(str(out_path))

    return written
