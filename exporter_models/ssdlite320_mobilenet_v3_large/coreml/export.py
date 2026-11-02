import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_model_module():
    pkg_init = _MODEL_DIR / "_model" / "__init__.py"
    module_name = f"_vendored_{_MODEL_DIR.name}_coreml_model"
    spec = importlib.util.spec_from_file_location(
        module_name, pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_model")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class _SsdliteWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        det = self.model([image])[0]
        return det["boxes"], det["scores"], det["labels"].float()


def export(config: dict, out_dir: str) -> str:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])
    precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
        variant["precision"]
    ]
    # generate_compile_specs defaults compute_unit to ALL. For a graph this small
    # that default is not obviously right: dispatching to the Neural Engine costs
    # the same whether the program does a lot of work or almost none, and ssdlite
    # is a 320x320 mobilenet. Swept on device rather than assumed; see
    # docs/ssdlite320-coreml-compute-units.md.
    compute_unit = getattr(ct.ComputeUnit, variant.get("compute_unit") or "ALL")

    m = _load_model_module()
    base_model = m.ssdlite320_mobilenet_v3_large(
        weights=m.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    ).eval()
    model = _SsdliteWrapper(base_model).eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    ep = torch_export(model, inputs)
    # iOS16, not the iOS18 used elsewhere: on an iPhone 16 the iOS17/18 programs run
    # at 6.7 ms against 3.4 ms, with the same mAP50-95 (22.25 vs 22.28 over 500 COCO
    # val images). RNE requires iOS 17, so this still loads everywhere.
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS16,
        compute_unit=compute_unit,
        compute_precision=precision,
    )
    program = to_edge_transform_and_lower(
        ep,
        partitioner=[
            CoreMLPartitioner(
                compile_specs=specs,
                skip_ops_for_coreml_delegation=["aten.select_scatter.default"],
            )
        ],
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
