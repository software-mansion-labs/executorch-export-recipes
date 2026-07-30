import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


_CRNN_WEIGHTS_URL = "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"


def _load_crnn():
    p = _MODEL_DIR / "_crnn.py"
    spec = importlib.util.spec_from_file_location("crnn_xnnpack_crnn", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crnn_xnnpack_crnn"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fetch_weights(url: str, member: str) -> Path:
    import urllib.request
    import zipfile

    cache = Path.home() / ".cache" / "exporter" / "crnn"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / member
    if not target.exists():
        zip_path = cache / "weights.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache)
        zip_path.unlink()
    return target


def export(config: dict, out_dir: str) -> str:
    from collections import OrderedDict

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    crnn = _load_crnn()
    Model = crnn.Model

    model = Model(num_class=97, input_channel=1, output_channel=256, hidden_size=256)

    state_dict = torch.load(
        str(_fetch_weights(_CRNN_WEIGHTS_URL, "english_g2.pth")), map_location="cpu", weights_only=False
    )
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:]
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.eval()

    widths = [64, 128, 256, 512]
    exported_programs = []
    for w in widths:
        sample_inputs = (torch.ones(1, 1, 64, w),)
        exported_programs.append(torch.export.export(model, sample_inputs))

    edge = to_edge_transform_and_lower(
        {
            "forward_64": exported_programs[0],
            "forward_128": exported_programs[1],
            "forward_256": exported_programs[2],
            "forward_512": exported_programs[3],
        },
        partitioner=[XnnpackPartitioner()],
    )
    exec_prog = edge.to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(exec_prog.buffer)
    return str(out_path)
