import importlib.util
import sys
from pathlib import Path

import torch
from torch.export import export as torch_export

from exporter.backends import lower

_HERE = Path(__file__).resolve().parent


def _load_model_module():
    pkg_init = _HERE / "_model" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "ssdlite_vendored_model", pkg_init,
        submodule_search_locations=[str(_HERE / "_model")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ssdlite_vendored_model"] = mod
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
    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    m = _load_model_module()
    base_model = m.ssdlite320_mobilenet_v3_large(
        weights=m.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    ).eval()
    model = _SsdliteWrapper(base_model).eval()
    inputs = (torch.randn(shape, dtype=torch.float32),)

    ep = torch_export(model, inputs)
    program = lower(ep, config["backend"])

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
