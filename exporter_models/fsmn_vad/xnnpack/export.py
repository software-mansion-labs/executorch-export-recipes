import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_vad_model():
    spec = importlib.util.spec_from_file_location("fsmn_vad_xnnpack_model", _MODEL_DIR / "_vad_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fsmn_vad_xnnpack_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from huggingface_hub import hf_hub_download
    from torch.export import Dim, export as torch_export

    vm = _load_vad_model()
    fwd = config["variant"]["methods"]["forward"]
    feat_shape = tuple(fwd["inputs"][0]["shape"])  # [1000, 512]

    model = vm.VAD(
        input_dim=400,
        input_affine_dim=140,
        fsmn_layers=4,
        linear_dim=250,
        proj_dim=128,
        lorder=20,
        rorder=0,
        lstride=1,
        rstride=1,
        output_affine_dim=140,
        output_dim=248,
        use_softmax=True,
    )

    ckpt = hf_hub_download("funasr/fsmn-vad", "model.pt")
    original_state_dict = torch.load(ckpt, map_location="cpu")

    prefix = "encoder."
    corrected_state_dict = {
        k[len(prefix):]: v
        for k, v in original_state_dict.items()
        if k.startswith(prefix)
    }

    model.load_state_dict(corrected_state_dict)
    model.eval()

    dummy = (torch.randn(*feat_shape, dtype=torch.float32),)
    dynamic_shapes = ({0: Dim("T", min=100, max=1000)},)

    with torch.no_grad():
        ep = torch_export(model, dummy, dynamic_shapes=dynamic_shapes)
        et_program = to_edge_transform_and_lower(
            ep,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et_program.buffer)
    return str(out_path)
