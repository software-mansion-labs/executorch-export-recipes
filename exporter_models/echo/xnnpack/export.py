"""Echo -> ExecuTorch XNNPACK (multi_pte), the programs react-native-executorch ships.

  reverb_xnnpack_8da8w.pte        backbone: prefill / step / polish (backbone.py), from the W8A8 checkpoint
  codec_decoder_xnnpack_fp16.pte  forward: codes [8, T] -> audio [T * 960], T 4-512 (decoder.py)
  codec_encoder_xnnpack_int8.pte  forward: wav [T, 960] -> codes [8, T], T 25-500 (encoder.py)

The weights come from Hugging Face (_echo/hub.py). The output is deterministic: each file is byte-identical to the
published one.
"""

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _module(path: Path):
    spec = importlib.util.spec_from_file_location(f"echo_xnnpack_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _program(name: str) -> bytes:
    if name.startswith("reverb_"):
        hub = _module(_HERE.parent / "_echo" / "hub.py")
        return _module(_HERE / "backbone.py").build(hub.checkpoint("int8a8"))
    if name.startswith("codec_decoder_"):
        return _module(_HERE / "decoder.py").build()
    if name.startswith("codec_encoder_"):
        return _module(_HERE / "encoder.py").build()
    raise ValueError(f"unknown Echo program for variant file '{name}'")


def export(config: dict, out_dir: str) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in config["variants"]:
        out_path = out_dir / variant["file"]
        out_path.write_bytes(_program(variant["file"]))
        written.append(str(out_path))
    return written
